import sys
from pathlib import Path
# Add the project root to the path if not already present
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.config.paths import get_project_paths, add_path_args

import os
import glob
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# MedMNISTC augmentation imports (ensure medmnistc is installed)
from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS

class MedMNISTAugTransform:
    def __init__(self, medmnist_aug):
        self.medmnist_aug = medmnist_aug
    
    def __call__(self, img):
        # Apply the medmnist augmentation
        result = self.medmnist_aug(img)
        # Check the type and convert if necessary
        if isinstance(result, np.ndarray):
            return Image.fromarray(result)
        elif isinstance(result, Image.Image):
            return result
        else:
            raise TypeError(f"Unexpected type from medmnist_aug: {type(result)}")

# Custom StainColorJitter class for histological stain augmentation
class StainColorJitter(object):
    """Applies stain-specific color jittering to tensor images."""
    M = torch.tensor([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]
    ])
    Minv = torch.inverse(M)
    eps = 1e-6

    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, P):
        assert P.dim() == 3 and P.size(0) == 3, "Input must be a 3-channel tensor"
        assert torch.all((P >= 0) & (P <= 1)), "Tensor values must be in [0, 1]"

        P_perm = P.permute(1, 2, 0)  # (H, W, 3)
        S = - (torch.log(255 * P_perm + self.eps)).matmul(self.Minv)  # (H, W, 3)
        alpha = 1 + (torch.rand(3) - 0.5) * 2 * self.sigma
        beta = (torch.rand(3) - 0.5) * 2 * self.sigma
        Sp = S * alpha + beta
        Pp = torch.exp(-Sp.matmul(self.M)) - self.eps
        Pp = Pp.permute(2, 0, 1) / 255  # (3, H, W)
        Pp = torch.clamp(Pp, 0.0, 1.0)
        return Pp

##############################
# 1. SET GLOBAL SEED & PATHS
##############################
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed globally
set_seed(42)

# Get project paths using the config module
paths = get_project_paths()

#########################################
# 2. CONFIGURATION, METRICS & AGGREGATION
#########################################
def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def compute_binary_metrics(true_labels, pred_probs, pred_classes, threshold=0.5):
    """Compute accuracy, sensitivity, specificity, F1 score, and AUC for binary classification.
       pred_probs are probabilities for the positive class.
    """
    binary_preds = (pred_probs >= threshold).astype(int)
    acc = accuracy_score(true_labels, binary_preds)
    f1 = f1_score(true_labels, binary_preds)
    auc_score = roc_auc_score(true_labels, pred_probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"accuracy": acc, "sensitivity": sensitivity, "specificity": specificity, "f1": f1, "auroc": auc_score}

def plot_roc_curves(roc_data_dict, title, save_path):
    """Plot ROC curves in a square figure."""
    plt.figure(figsize=(8, 8))
    for label, (fpr, tpr, auc_score) in roc_data_dict.items():
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC plot saved to {save_path}")

# Aggregation strategies for grouping tile-level predictions
aggregation_strategies = {
    'mean': lambda x: np.mean(x),
    'median': lambda x: np.median(x),
    'top_k_mean_10': lambda x: np.mean(np.sort(x)[-max(1, int(len(x)*0.1)):]) if len(x) > 0 else 0,
    'top_k_mean_20': lambda x: np.mean(np.sort(x)[-max(1, int(len(x)*0.2)):]) if len(x) > 0 else 0,
    'top_k_mean_30': lambda x: np.mean(np.sort(x)[-max(1, int(len(x)*0.3)):]) if len(x) > 0 else 0,
}

def aggregate_by_group(true_labels, pred_probs, groups, agg_func):
    """Aggregate predictions by group using the given aggregation function."""
    unique_groups = np.unique(groups)
    agg_true = []
    agg_probs = []
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        # Assume that all tiles in a group share the same true label
        group_true = np.mean(true_labels[idx])
        group_prob = agg_func(pred_probs[idx])
        agg_true.append(group_true)
        agg_probs.append(group_prob)
    return np.array(agg_true), np.array(agg_probs)

def optimize_threshold(agg_true, agg_probs):
    """Search over thresholds (0–1) to find the one maximizing balanced accuracy."""
    best_threshold = 0.5
    best_bal_acc = -np.inf
    best_metrics = {}
    for thresh in np.linspace(0, 1, 101):
        preds = (agg_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(agg_true, preds).ravel()
        sensitivity = tp / (tp+fn) if (tp+fn) > 0 else 0
        specificity = tn / (tn+fp) if (tn+fp) > 0 else 0
        bal_acc = (sensitivity + specificity) / 2
        f1 = f1_score(agg_true, preds)
        acc = accuracy_score(agg_true, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1': f1,
                'balanced_acc': bal_acc,
                'accuracy': acc,
                'auc': roc_auc_score(agg_true, agg_probs)
            }
    return best_threshold, best_metrics

#################################
# 3. DATASET CLASSES (WITH GROUPS)
#################################

# Inflammation dataset – grouping by slide_name (existing splits are used)
class HETileDataset(Dataset):
    def __init__(self, csv_path: Path, tiles_dir: Path, transform=None):
        """
        csv_path: CSV file (e.g. train_HE.csv, val_HE.csv, or test_HE.csv)
        tiles_dir: directory with PNG tile images
        """
        self.transform = transform
        self.tiles_dir = str(tiles_dir)
        self.data = pd.read_csv(csv_path)
        # Filter for HE tiles if necessary
        self.data = self.data[self.data["stain"].str.upper() == "HE"]
        # Map inflammation_status to binary label: inflamed -> 1, noninflamed -> 0
        self.data['label'] = self.data['inflammation_status'].apply(lambda x: 1 if x.lower() == 'inflamed' else 0)
        self.samples = []
        for _, row in self.data.iterrows():
            slide_name = row['slide_name']
            label = row['label']
            pattern = os.path.join(self.tiles_dir, f"*{slide_name}*.png")
            files = glob.glob(pattern)
            if not files:
                print(f"Warning: No files found for slide {slide_name} using pattern {pattern}")
            for f in files:
                if self._is_valid_tile(Path(f)):
                    self.samples.append((f, label, slide_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, group_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), group_id

    def _is_valid_tile(self, image_path: Path) -> bool:
        """Validate tile contains sufficient tissue content."""
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
                white_pixels = np.sum(img_array > 240) / img_array.size
                return white_pixels < 0.9
        except Exception as e:
            print(f"Error validating tile {image_path}: {e}")
            return False

# Tissue classification dataset using provided splits.
class TissueSplitDataset(Dataset):
    def __init__(self, split_csv: Path, tissue_csv: Path, tiles_dir: Path, transform=None):
        """
        split_csv: CSV file from splits folder (e.g., train_HE.csv, val_HE.csv, or test_HE.csv)
        tissue_csv: CSV file with tissue labels (tissue_types.csv)
        tiles_dir: directory with PNG tile images
        """
        self.tiles_dir = str(tiles_dir)
        self.transform = transform
        split_df = pd.read_csv(split_csv)
        split_df = split_df[split_df["stain"].str.upper() == "HE"]
        tissue_df = pd.read_csv(tissue_csv)
        self.data = pd.merge(split_df, tissue_df, on="slide_name", how="inner")
        mapping = {"antrum": 0, "corpus": 1}
        self.data['label'] = self.data['tissue_type'].str.lower().map(mapping)
        self.data = self.data.dropna(subset=["label"])
        self.data['label'] = self.data['label'].astype(int)
        self.samples = []
        for _, row in self.data.iterrows():
            slide_name = row['slide_name']
            label = row['label']
            pattern = os.path.join(self.tiles_dir, f"*{slide_name}*.png")
            files = glob.glob(pattern)
            if not files:
                print(f"Warning: No files found for slide {slide_name} using pattern {pattern}")
            for f in files:
                if self._is_valid_tile(Path(f)):
                    self.samples.append((f, label, slide_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, group_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), group_id

    def _is_valid_tile(self, image_path: Path) -> bool:
        """Validate tile contains sufficient tissue content."""
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
                white_pixels = np.sum(img_array > 240) / img_array.size
                return white_pixels < 0.9
        except Exception as e:
            print(f"Error validating tile {image_path}: {e}")
            return False

####################################
# 4. TRANSFORMS & MODEL BUILDING
####################################
def build_transforms(common_cfg, task_cfg, mode="training",
                     use_medmnist_aug=False, use_stain_color_jitter=False, only_normalization=False):
    """Build a torchvision transform pipeline."""
    normalization = common_cfg["transforms"]["normalization"]
    mean = normalization["mean"]
    std = normalization["std"]
    
    transform_list = []
    
    if only_normalization:
        val_cfg = common_cfg["transforms"]["validation"]
        transform_list.append(transforms.Resize(val_cfg["resize"]))
        transform_list.append(transforms.CenterCrop(val_cfg["center_crop"]))
    else:
        if mode == "training":
            if use_medmnist_aug:
                medmnist_aug = AugMedMNISTC(CORRUPTIONS_DS["pathmnist"])
                transform_list.append(MedMNISTAugTransform(medmnist_aug))
            rr_crop_cfg = task_cfg["transforms"]["training"]["random_resized_crop"]
            transform_list.append(transforms.RandomResizedCrop(size=rr_crop_cfg["size"], scale=tuple(rr_crop_cfg["scale"])))
            if task_cfg["transforms"]["training"].get("random_horizontal_flip", False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if task_cfg["transforms"]["training"].get("random_vertical_flip", False):
                transform_list.append(transforms.RandomVerticalFlip())
            rr_cfg = task_cfg["transforms"]["training"].get("random_rotation", {})
            if rr_cfg.get("enabled", False):
                transform_list.append(transforms.RandomRotation(degrees=rr_cfg.get("degrees", 0)))
            rb_cfg = task_cfg["transforms"]["training"].get("random_blur", {})
            if rb_cfg.get("enabled", False):
                kernel_size = rb_cfg.get("kernel_size", 3)
                transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size))
            ra_cfg = task_cfg["transforms"]["training"].get("random_affine", {})
            if ra_cfg.get("enabled", False):
                transform_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        else:
            val_cfg = common_cfg["transforms"]["validation"]
            transform_list.append(transforms.Resize(val_cfg["resize"]))
            transform_list.append(transforms.CenterCrop(val_cfg["center_crop"]))
    
    transform_list.append(transforms.ToTensor())
    if mode == "training" and use_stain_color_jitter:
        transform_list.append(StainColorJitter(sigma=0.05))
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)

def build_densenet121_model(task_config, num_classes=1):
    """Build DenseNet121. If num_classes==1, uses BCEWithLogitsLoss; otherwise CrossEntropyLoss."""
    arch_cfg = task_config["architecture"]
    model = torchvision.models.densenet121(weights="IMAGENET1K_V1")
    if arch_cfg.get("feature_extraction", False):
        for param in model.parameters():
            param.requires_grad = False
    dropout_rate = task_config.get("dropout_rate", 0.0)
    num_features = model.classifier.in_features
    if dropout_rate > 0:
        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        classifier = nn.Linear(num_features, num_classes)
    model.classifier = classifier
    return model

#####################################
# 5. TRAINING & EVALUATION FUNCTIONS
#####################################
def train_model(model, dataloaders, device, config_task, num_classes=1):
    num_epochs = config_task["epochs"]
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    
    optim_cfg = config_task["optimizer"]
    if optim_cfg["name"].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), 
                                lr=optim_cfg["learning_rate"],
                                weight_decay=optim_cfg["weight_decay"])
    elif optim_cfg["name"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=optim_cfg["learning_rate"],
                              momentum=optim_cfg.get("momentum", 0.9),
                              weight_decay=optim_cfg["weight_decay"])
    else:
        raise ValueError("Unsupported optimizer")
    
    scheduler = None
    sched_cfg = config_task.get("scheduler", {})
    if sched_cfg.get("enabled", False):
        if sched_cfg["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=sched_cfg.get("T_max", num_epochs))
    
    early_stop_cfg = config_task.get("early_stopping", {})
    early_stopping_enabled = early_stop_cfg.get("enabled", False)
    patience = early_stop_cfg.get("patience", 10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    
    history = {"train_loss": [], "val_loss": []}
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            if num_classes == 1:
                labels = labels.to(device).unsqueeze(1).float()
            else:
                labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_clip = config_task.get("gradient_clipping", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloaders["train"].dataset)
        history["train_loss"].append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                if len(batch) == 3:
                    inputs, labels, _ = batch
                else:
                    inputs, labels = batch
                inputs = inputs.to(device)
                if num_classes == 1:
                    labels = labels.to(device).unsqueeze(1).float()
                else:
                    labels = labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_loss / len(dataloaders["val"].dataset)
        history["val_loss"].append(epoch_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Train loss: {epoch_loss:.4f} | Val loss: {epoch_val_loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
        if early_stopping_enabled:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                best_model_wts = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered")
                    if best_model_wts is not None:
                        model.load_state_dict(best_model_wts)
                    break
    return model, history

def evaluate_model_with_groups(model, dataloader, device, num_classes=1):
    """Evaluate the model and collect predictions, true labels, and grouping keys."""
    model.eval()
    all_true = []
    all_probs = []
    all_preds = []
    groups = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if len(batch) == 3:
                inputs, labels, group_ids = batch
            else:
                inputs, labels = batch
                group_ids = None
            inputs = inputs.to(device)
            outputs = model(inputs)
            if num_classes == 1:
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
            else:
                probs_full = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs_full, axis=1)
                probs = probs_full[:, 1]
            all_true.extend(labels.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            if group_ids is not None:
                groups.extend(group_ids)
    return np.array(all_true), np.array(all_probs), np.array(all_preds), (np.array(groups) if len(groups) > 0 else None)

#####################################
# 6. MAIN EXPERIMENT LOOP & RESULTS
#####################################
if __name__ == "__main__":
    # Build paths for configuration and data using project paths
    config_path = paths["CONFIG_DIR"] / "model_config.yaml"
    tiles_dir = paths["PROCESSED_DIR"] / "tiles"
    
    # Inflammation splits from the splits folder
    infl_train_csv = paths["SPLITS_DIR"] / "seed_42" / "train_HE.csv"
    infl_val_csv   = paths["SPLITS_DIR"] / "seed_42" / "val_HE.csv"
    infl_test_csv  = paths["SPLITS_DIR"] / "seed_42" / "test_HE.csv"
    infl_test_scanner2_csv = paths["SPLITS_DIR"] / "seed_42" / "test_scanner2_HE.csv"
    
    # Tissue splits (using HE splits and tissue_types.csv)
    tissue_train_csv = paths["SPLITS_DIR"] / "seed_42" / "train_HE.csv"
    tissue_val_csv   = paths["SPLITS_DIR"] / "seed_42" / "val_HE.csv"
    tissue_test_csv  = paths["SPLITS_DIR"] / "seed_42" / "test_HE.csv"
    tissue_test_scanner2_csv = paths["SPLITS_DIR"] / "seed_42" / "test_scanner2_HE.csv"
    tissue_types_csv = paths["METRICS_DIR"] / "tissue_types.csv"
    
    # Load YAML configuration
    config = load_config(config_path)
    common_cfg = config["common"]
    
    device = torch.device(common_cfg.get("device", "cpu"))
    
    # Augmentation variants to test
    augmentation_variants = {
        "Medmnist": {"use_medmnist_aug": True, "use_stain_color_jitter": False, "only_normalization": False},
        "ColorJitter": {"use_medmnist_aug": True, "use_stain_color_jitter": True, "only_normalization": False},
        "NormalizationOnly": {"only_normalization": True},
        "ModelConfig": {"use_medmnist_aug": False, "use_stain_color_jitter": False, "only_normalization": False}
    }
    
    # Dictionaries to collect ROC data for plotting (separate for each scanner)
    roc_infl_tile = {}
    roc_infl_tile_scanner2 = {}
    roc_infl_slide = {}
    roc_infl_slide_scanner2 = {}
    roc_tissue_tile = {}
    roc_tissue_tile_scanner2 = {}
    roc_tissue_particle = {}
    roc_tissue_particle_scanner2 = {}
    
    # Store results for file output
    results_infl = {}
    results_tissue = {}
    
    #############################
    # Inflammation Classification
    #############################
    print("\n===== Running Inflammation Classification Experiments =====")
    infl_task_cfg = config["inflammation"]["densenet121"]
    num_classes_infl = 1  # Binary classification
    
    for variant_name, aug_params in augmentation_variants.items():
        print(f"\n--- Inflammation Variant: {variant_name} ---")
        train_transform = build_transforms(common_cfg, infl_task_cfg, mode="training",
                                           use_medmnist_aug=aug_params.get("use_medmnist_aug", False),
                                           use_stain_color_jitter=aug_params.get("use_stain_color_jitter", False),
                                           only_normalization=aug_params.get("only_normalization", False))
        val_transform = build_transforms(common_cfg, infl_task_cfg, mode="validation")
        
        train_dataset = HETileDataset(infl_train_csv, tiles_dir, transform=train_transform)
        val_dataset   = HETileDataset(infl_val_csv, tiles_dir, transform=val_transform)
        test_dataset  = HETileDataset(infl_test_csv, tiles_dir, transform=val_transform)
        test_scanner2_dataset = HETileDataset(infl_test_scanner2_csv, tiles_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=infl_task_cfg["batch_size"], shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=infl_task_cfg["batch_size"], shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=infl_task_cfg["batch_size"], shuffle=False, num_workers=4)
        test_scanner2_loader = DataLoader(test_scanner2_dataset, batch_size=infl_task_cfg["batch_size"], shuffle=False, num_workers=4)
        
        dataloaders = {"train": train_loader, "val": val_loader}
        
        model = build_densenet121_model(infl_task_cfg, num_classes=num_classes_infl)
        model, history = train_model(model, dataloaders, device, infl_task_cfg, num_classes=num_classes_infl)
        
        # TILE-LEVEL EVALUATION on Test Set (test_HE.csv)
        true_tile, probs_tile, preds_tile, groups_tile = evaluate_model_with_groups(model, test_loader, device, num_classes=num_classes_infl)
        tile_metrics = compute_binary_metrics(true_tile, probs_tile, preds_tile)
        roc_infl_tile[variant_name] = roc_curve(true_tile, probs_tile)[:2] + (roc_auc_score(true_tile, probs_tile),)
        
        # TILE-LEVEL EVALUATION on Scanner2 Test Set (test_scanner2_HE.csv)
        true_tile_s2, probs_tile_s2, preds_tile_s2, groups_tile_s2 = evaluate_model_with_groups(model, test_scanner2_loader, device, num_classes=num_classes_infl)
        tile_metrics_s2 = compute_binary_metrics(true_tile_s2, probs_tile_s2, preds_tile_s2)
        roc_infl_tile_scanner2[variant_name] = roc_curve(true_tile_s2, probs_tile_s2)[:2] + (roc_auc_score(true_tile_s2, probs_tile_s2),)
        
        # AGGREGATED EVALUATION (slide-level) for test_HE.csv
        true_val, probs_val, preds_val, groups_val = evaluate_model_with_groups(model, val_loader, device, num_classes=num_classes_infl)
        best_strategy = None
        best_threshold = 0.5
        best_val_bal_acc = -np.inf
        best_val_metrics = {}
        for strat_name, strat_func in aggregation_strategies.items():
            agg_true_val, agg_probs_val = aggregate_by_group(true_val, probs_val, groups_val, strat_func)
            thresh, metrics_opt = optimize_threshold(agg_true_val, agg_probs_val)
            if metrics_opt['balanced_acc'] > best_val_bal_acc:
                best_val_bal_acc = metrics_opt['balanced_acc']
                best_strategy = strat_name
                best_threshold = thresh
                best_val_metrics = metrics_opt
        true_test, probs_test, preds_test, groups_test = evaluate_model_with_groups(model, test_loader, device, num_classes=num_classes_infl)
        agg_true_test, agg_probs_test = aggregate_by_group(true_test, probs_test, groups_test, aggregation_strategies[best_strategy])
        agg_preds_test = (agg_probs_test >= best_threshold).astype(int)
        agg_metrics = compute_binary_metrics(agg_true_test, agg_probs_test, agg_preds_test, threshold=best_threshold)
        roc_infl_slide[variant_name] = roc_curve(agg_true_test, agg_probs_test)[:2] + (roc_auc_score(agg_true_test, agg_probs_test),)
        
        # AGGREGATED EVALUATION (slide-level) for test_scanner2_HE.csv
        true_test_s2, probs_test_s2, preds_test_s2, groups_test_s2 = evaluate_model_with_groups(model, test_scanner2_loader, device, num_classes=num_classes_infl)
        agg_true_test_s2, agg_probs_test_s2 = aggregate_by_group(true_test_s2, probs_test_s2, groups_test_s2, aggregation_strategies[best_strategy])
        agg_preds_test_s2 = (agg_probs_test_s2 >= best_threshold).astype(int)
        agg_metrics_s2 = compute_binary_metrics(agg_true_test_s2, agg_probs_test_s2, agg_preds_test_s2, threshold=best_threshold)
        roc_infl_slide_scanner2[variant_name] = roc_curve(agg_true_test_s2, agg_probs_test_s2)[:2] + (roc_auc_score(agg_true_test_s2, agg_probs_test_s2),)
        
        results_infl[variant_name] = {
            "tile_metrics": tile_metrics,
            "tile_metrics_scanner2": tile_metrics_s2,
            "optimal_aggregation": {
                "best_strategy": best_strategy,
                "best_metrics": best_val_metrics,
                "threshold": best_threshold,
                "aggregated_test_metrics": agg_metrics,
                "aggregated_test_metrics_scanner2": agg_metrics_s2
            }
        }
    
    #########################
    # Tissue Classification
    #########################
    print("\n===== Running Tissue Classification Experiments =====")
    tissue_task_cfg = config["tissue"]["densenet121"]
    num_classes_tissue = 2  # Using CrossEntropyLoss
    
    train_dataset = TissueSplitDataset(tissue_train_csv, tissue_types_csv, tiles_dir, transform=build_transforms(common_cfg, tissue_task_cfg, mode="training"))
    val_dataset   = TissueSplitDataset(tissue_val_csv, tissue_types_csv, tiles_dir, transform=build_transforms(common_cfg, tissue_task_cfg, mode="validation"))
    test_dataset  = TissueSplitDataset(tissue_test_csv, tissue_types_csv, tiles_dir, transform=build_transforms(common_cfg, tissue_task_cfg, mode="validation"))
    test_scanner2_dataset = TissueSplitDataset(tissue_test_scanner2_csv, tissue_types_csv, tiles_dir, transform=build_transforms(common_cfg, tissue_task_cfg, mode="validation"))
    
    for variant_name, aug_params in augmentation_variants.items():
        print(f"\n--- Tissue Variant: {variant_name} ---")
        train_transform = build_transforms(common_cfg, tissue_task_cfg, mode="training",
                                           use_medmnist_aug=aug_params.get("use_medmnist_aug", False),
                                           use_stain_color_jitter=aug_params.get("use_stain_color_jitter", False),
                                           only_normalization=aug_params.get("only_normalization", False))
        val_transform = build_transforms(common_cfg, tissue_task_cfg, mode="validation")
        
        train_dataset.transform = train_transform
        val_dataset.transform = val_transform
        test_dataset.transform = val_transform
        test_scanner2_dataset.transform = val_transform
        
        train_loader = DataLoader(train_dataset, batch_size=tissue_task_cfg["batch_size"], shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=tissue_task_cfg["batch_size"], shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=tissue_task_cfg["batch_size"], shuffle=False, num_workers=4)
        test_scanner2_loader = DataLoader(test_scanner2_dataset, batch_size=tissue_task_cfg["batch_size"], shuffle=False, num_workers=4)
        
        dataloaders = {"train": train_loader, "val": val_loader}
        
        model = build_densenet121_model(tissue_task_cfg, num_classes=num_classes_tissue)
        model, history = train_model(model, dataloaders, device, tissue_task_cfg, num_classes=num_classes_tissue)
        
        # TILE-LEVEL EVALUATION on Test Set (test_HE.csv)
        true_tile, probs_tile, preds_tile, groups_tile = evaluate_model_with_groups(model, test_loader, device, num_classes=num_classes_tissue)
        tile_metrics = compute_binary_metrics(true_tile, probs_tile, preds_tile)
        roc_tissue_tile[variant_name] = roc_curve(true_tile, probs_tile)[:2] + (roc_auc_score(true_tile, probs_tile),)
        
        # TILE-LEVEL EVALUATION on Scanner2 Test Set (test_scanner2_HE.csv)
        true_tile_s2, probs_tile_s2, preds_tile_s2, groups_tile_s2 = evaluate_model_with_groups(model, test_scanner2_loader, device, num_classes=num_classes_tissue)
        tile_metrics_s2 = compute_binary_metrics(true_tile_s2, probs_tile_s2, preds_tile_s2)
        roc_tissue_tile_scanner2[variant_name] = roc_curve(true_tile_s2, probs_tile_s2)[:2] + (roc_auc_score(true_tile_s2, probs_tile_s2),)
        
        # AGGREGATED EVALUATION (particle-level) for test_HE.csv
        true_val, probs_val, preds_val, groups_val = evaluate_model_with_groups(model, val_loader, device, num_classes=num_classes_tissue)
        best_strategy = None
        best_threshold = 0.5
        best_val_bal_acc = -np.inf
        best_val_metrics = {}
        for strat_name, strat_func in aggregation_strategies.items():
            agg_true_val, agg_probs_val = aggregate_by_group(true_val, probs_val, groups_val, strat_func)
            thresh, metrics_opt = optimize_threshold(agg_true_val, agg_probs_val)
            if metrics_opt['balanced_acc'] > best_val_bal_acc:
                best_val_bal_acc = metrics_opt['balanced_acc']
                best_strategy = strat_name
                best_threshold = thresh
                best_val_metrics = metrics_opt
        true_test, probs_test, preds_test, groups_test = evaluate_model_with_groups(model, test_loader, device, num_classes=num_classes_tissue)
        agg_true_test, agg_probs_test = aggregate_by_group(true_test, probs_test, groups_test, aggregation_strategies[best_strategy])
        agg_preds_test = (agg_probs_test >= best_threshold).astype(int)
        agg_metrics = compute_binary_metrics(agg_true_test, agg_probs_test, agg_preds_test, threshold=best_threshold)
        roc_tissue_particle[variant_name] = roc_curve(agg_true_test, agg_probs_test)[:2] + (roc_auc_score(agg_true_test, agg_probs_test),)
        
        # AGGREGATED EVALUATION (particle-level) for test_scanner2_HE.csv
        true_test_s2, probs_test_s2, preds_test_s2, groups_test_s2 = evaluate_model_with_groups(model, test_scanner2_loader, device, num_classes=num_classes_tissue)
        agg_true_test_s2, agg_probs_test_s2 = aggregate_by_group(true_test_s2, probs_test_s2, groups_test_s2, aggregation_strategies[best_strategy])
        agg_preds_test_s2 = (agg_probs_test_s2 >= best_threshold).astype(int)
        agg_metrics_s2 = compute_binary_metrics(agg_true_test_s2, agg_probs_test_s2, agg_preds_test_s2, threshold=best_threshold)
        roc_tissue_particle_scanner2[variant_name] = roc_curve(agg_true_test_s2, agg_probs_test_s2)[:2] + (roc_auc_score(agg_true_test_s2, agg_probs_test_s2),)
        
        results_tissue[variant_name] = {
            "tile_metrics": tile_metrics,
            "tile_metrics_scanner2": tile_metrics_s2,
            "optimal_aggregation": {
                "best_strategy": best_strategy,
                "best_metrics": best_val_metrics,
                "threshold": best_threshold,
                "aggregated_test_metrics": agg_metrics,
                "aggregated_test_metrics_scanner2": agg_metrics_s2
            }
        }
    
    #####################################
    # Generate ROC Plots (8 Figures Total - Separate for Each Scanner)
    #####################################
    # Inflammation - Primary Scanner
    plot_roc_curves(roc_infl_tile, title="Inflammation - Tile-Level ROC (Primary Scanner)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_inflammation_tile_primary.png"))
    plot_roc_curves(roc_infl_slide, title="Inflammation - Slide-Level ROC (Primary Scanner)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_inflammation_slide_primary.png"))
    
    # Inflammation - Scanner2
    plot_roc_curves(roc_infl_tile_scanner2, title="Inflammation - Tile-Level ROC (Scanner2)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_inflammation_tile_scanner2.png"))
    plot_roc_curves(roc_infl_slide_scanner2, title="Inflammation - Slide-Level ROC (Scanner2)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_inflammation_slide_scanner2.png"))
    
    # Tissue - Primary Scanner
    plot_roc_curves(roc_tissue_tile, title="Tissue - Tile-Level ROC (Primary Scanner)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_tissue_tile_primary.png"))
    plot_roc_curves(roc_tissue_particle, title="Tissue - Particle-Level ROC (Primary Scanner)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_tissue_particle_primary.png"))
    
    # Tissue - Scanner2
    plot_roc_curves(roc_tissue_tile_scanner2, title="Tissue - Tile-Level ROC (Scanner2)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_tissue_tile_scanner2.png"))
    plot_roc_curves(roc_tissue_particle_scanner2, title="Tissue - Particle-Level ROC (Scanner2)",
                    save_path=str(paths["FIGURES_DIR"] / "roc_tissue_particle_scanner2.png"))
    
    #####################################
    # Save Detailed Results to Files
    #####################################
    metric_types = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auroc']
    
    # Inflammation results file
    for variant, metrics in results_infl.items():
        filename = str(paths["RESULTS_DIR"] / f"results_inflammation_{variant}.txt")
        with open(filename, "w") as f:
            f.write(f"INFLAMMATION RESULTS - Variant: {variant}\n")
            f.write("="*40 + "\n\n")
            
            # Tile-Level Metrics (Primary Scanner - test_HE.csv)
            f.write("TEST DATASET - TILE LEVEL METRICS (Primary Scanner)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                f.write(f"{m.capitalize()}: {metrics['tile_metrics'].get(m, 'N/A'):.3f}\n")
            
            # Tile-Level Metrics (Scanner2 - test_scanner2_HE.csv)
            f.write("\nTEST DATASET - TILE LEVEL METRICS (Scanner2)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                f.write(f"{m.capitalize()}: {metrics['tile_metrics_scanner2'].get(m, 'N/A'):.3f}\n")
            
            # Optimal Aggregation Strategy (Validation Set)
            if 'optimal_aggregation' in metrics:
                agg = metrics['optimal_aggregation']
                f.write("\nOPTIMAL AGGREGATION STRATEGY (Validation Set)\n")
                f.write("-" * 45 + "\n")
                f.write(f"Strategy: {agg.get('best_strategy', 'unknown')}\n")
                best_metrics = agg.get('best_metrics', {})
                f.write(f"Threshold: {best_metrics.get('threshold', 0.5):.3f}\n")
                f.write(f"Sensitivity: {best_metrics.get('sensitivity', 0):.2%}\n")
                f.write(f"Specificity: {best_metrics.get('specificity', 0):.2%}\n")
                f.write(f"F1 Score: {best_metrics.get('f1', 0):.2%}\n")
                f.write(f"Balanced Accuracy: {best_metrics.get('balanced_acc', 0):.2%}\n")
                f.write(f"AUC: {best_metrics.get('auc', 0):.2%}\n")
            
            # Aggregated Test Metrics (Primary Scanner - test_HE.csv)
            f.write("\nAGGREGATED TEST METRICS (Slide-Level, Primary Scanner)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                agg_val = metrics['optimal_aggregation'].get('aggregated_test_metrics', {}).get(m, 'N/A')
                f.write(f"{m.capitalize()}: {agg_val:.3f}\n")
            
            # Aggregated Test Metrics (Scanner2 - test_scanner2_HE.csv)
            f.write("\nAGGREGATED TEST METRICS (Slide-Level, Scanner2)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                agg_val = metrics['optimal_aggregation'].get('aggregated_test_metrics_scanner2', {}).get(m, 'N/A')
                f.write(f"{m.capitalize()}: {agg_val:.3f}\n")
        
        print(f"Inflammation results saved to {filename}")
    
    # Tissue results file
    for variant, metrics in results_tissue.items():
        filename = str(paths["RESULTS_DIR"] / f"results_tissue_{variant}.txt")
        with open(filename, "w") as f:
            f.write(f"TISSUE RESULTS - Variant: {variant}\n")
            f.write("="*40 + "\n\n")
            
            # Tile-Level Metrics (Primary Scanner - test_HE.csv)
            f.write("TEST DATASET - TILE LEVEL METRICS (Primary Scanner)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                f.write(f"{m.capitalize()}: {metrics['tile_metrics'].get(m, 'N/A'):.3f}\n")
            
            # Tile-Level Metrics (Scanner2 - test_scanner2_HE.csv)
            f.write("\nTEST DATASET - TILE LEVEL METRICS (Scanner2)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                f.write(f"{m.capitalize()}: {metrics['tile_metrics_scanner2'].get(m, 'N/A'):.3f}\n")
            
            # Optimal Aggregation Strategy (Validation Set)
            if 'optimal_aggregation' in metrics:
                agg = metrics['optimal_aggregation']
                f.write("\nOPTIMAL AGGREGATION STRATEGY (Validation Set)\n")
                f.write("-" * 45 + "\n")
                f.write(f"Strategy: {agg.get('best_strategy', 'unknown')}\n")
                best_metrics = agg.get('best_metrics', {})
                f.write(f"Threshold: {best_metrics.get('threshold', 0.5):.3f}\n")
                f.write(f"Sensitivity: {best_metrics.get('sensitivity', 0):.2%}\n")
                f.write(f"Specificity: {best_metrics.get('specificity', 0):.2%}\n")
                f.write(f"F1 Score: {best_metrics.get('f1', 0):.2%}\n")
                f.write(f"Balanced Accuracy: {best_metrics.get('balanced_acc', 0):.2%}\n")
                f.write(f"AUC: {best_metrics.get('auc', 0):.2%}\n")
            
            # Aggregated Test Metrics (Primary Scanner - test_HE.csv)
            f.write("\nAGGREGATED TEST METRICS (Particle-Level, Primary Scanner)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                agg_val = metrics['optimal_aggregation'].get('aggregated_test_metrics', {}).get(m, 'N/A')
                f.write(f"{m.capitalize()}: {agg_val:.3f}\n")
            
            # Aggregated Test Metrics (Scanner2 - test_scanner2_HE.csv)
            f.write("\nAGGREGATED TEST METRICS (Particle-Level, Scanner2)\n")
            f.write("-"*45 + "\n")
            for m in metric_types:
                agg_val = metrics['optimal_aggregation'].get('aggregated_test_metrics_scanner2', {}).get(m, 'N/A')
                f.write(f"{m.capitalize()}: {agg_val:.3f}\n")
        
        print(f"Tissue results saved to {filename}")