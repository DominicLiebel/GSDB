import sys
from pathlib import Path
import logging
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torch.serialization
import json
from datetime import datetime

# Add numpy scalar to the list of safe globals
torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config.paths import get_project_paths, add_path_args
from src.models.dataset import HistologyDataset
from src.models.model_utils import get_transforms, load_model
import src.models.training_utils as training_utils
import src.models.metrics_utils as metrics_utils
import src.models.evaluate as evaluate

# MedMNISTC augmentation imports
try:
    from medmnistc.augmentation import AugMedMNISTC
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    MEDMNISTC_AVAILABLE = True
except ImportError:
    MEDMNISTC_AVAILABLE = False
    logging.warning("medmnistc not available; Medmnist variant will use default transforms")

# Custom transform classes
class MedMNISTAugTransform:
    """Wraps MedMNISTC augmentation for torchvision compatibility."""
    def __init__(self, medmnist_aug):
        self.medmnist_aug = medmnist_aug
    
    def __call__(self, img):
        result = self.medmnist_aug(img)
        if isinstance(result, np.ndarray):
            return Image.fromarray(result)
        elif isinstance(result, Image.Image):
            return result
        else:
            raise TypeError(f"Unexpected type from medmnist_aug: {type(result)}")

class StainColorJitter:
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

def setup_logging(output_dir: Path) -> None:
    """Configure logging for both file and console output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs subdirectory
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / 'experiment.log'
    
    # Configure logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Logging configured. Log file: {log_file}")

def load_config(config_path: Path, task: str) -> dict:
    """Load configuration from model_config.yaml with task-specific settings."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    common_config = config.get('common', {})
    task_config = config[task].get('densenet121', {})
    result_config = common_config.copy()
    result_config.update(task_config)
    result_config['task'] = task
    result_config['selected_model'] = 'densenet121'
    
    if task == 'inflammation':
        result_config['batch_size'] = 32
        result_config['dropout_rate'] = 0.16141698206729396
        result_config['epochs'] = 50
        result_config['optimizer']['name'] = 'AdamW'
        result_config['optimizer']['learning_rate'] = 0.0001776588643563922
        result_config['optimizer']['weight_decay'] = 3.43776260302271e-05
        result_config['pos_weight'] = 0.4129621502243413
        result_config['scheduler'] = {'enabled': True, 'name': 'CosineAnnealingLR', 'T_max': 30}
        result_config['early_stopping'] = {'enabled': True, 'patience': 10}
    elif task == 'tissue':
        result_config['batch_size'] = 64
        result_config['dropout_rate'] = 0.0612284423470278
        result_config['epochs'] = 50
        result_config['optimizer']['name'] = 'AdamW'
        result_config['optimizer']['learning_rate'] = 0.00018943422706671847
        result_config['optimizer']['weight_decay'] = 1.0775930465603428e-05
        result_config['pos_weight'] = 0.654171439017536
        result_config['scheduler'] = {'enabled': True, 'name': 'CosineAnnealingLR', 'T_max': 50}
    
    logging.info(f"Loaded config for {task}: {result_config}")
    return result_config

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_binary_metrics(true_labels, pred_probs, threshold=0.5):
    """Compute accuracy, sensitivity, specificity, F1 score, and AUC for binary classification."""
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
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.axis("square")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"ROC plot saved to {save_path}")

def train_and_evaluate(args, paths):
    """Train and evaluate DenseNet121 with variants, hardcoded thresholds, and aggregation."""
    config = load_config(paths["CONFIG_DIR"] / "model_config.yaml", args.task)
    
    # Use the same seed as in evaluate.py
    set_seed(42)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        logging.info(f"Using {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        logging.info("No GPUs available, using CPU")

    augmentation_variants = {
        "Medmnist": {"use_medmnist_aug": True, "use_stain_color_jitter": False, "only_normalization": False, "use_model_config": False},
        "ColorJitter": {"use_medmnist_aug": False, "use_stain_color_jitter": True, "only_normalization": False, "use_model_config": False},
        "ModelConfig": {"use_medmnist_aug": False, "use_stain_color_jitter": False, "only_normalization": False, "use_model_config": True},
        "Medmnist_ColorJitter": {"use_medmnist_aug": True, "use_stain_color_jitter": True, "only_normalization": False, "use_model_config": False},
        "All": {"use_medmnist_aug": True, "use_stain_color_jitter": True, "only_normalization": False, "use_model_config": True},
        "NormalizationOnly": {"use_medmnist_aug": False, "use_stain_color_jitter": False, "only_normalization": True, "use_model_config": False}   
    }
    
    if args.variant:
        if args.variant not in augmentation_variants:
            raise ValueError(f"Invalid variant: {args.variant}. Choices are {list(augmentation_variants.keys())}")
        augmentation_variants = {args.variant: augmentation_variants[args.variant]}
        
        # If we're using ModelConfig variant, use the exact same evaluation approach as evaluate.py
        if args.variant == "ModelConfig":
            logging.info("ModelConfig variant selected - using evaluate.py approach for exact same results")
            
            # Map test_split to scanner name for directory naming (same as in evaluate.py)
            scanner_name = {
                'test': 'scanner1',
                'test_scanner2': 'scanner2'
            }.get(args.test_split, args.test_split)
            
            # Create evaluation directory with similar naming convention to evaluate.py
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths["EXPERIMENTS_DIR"] / f'{args.task}_{scanner_name}_densenet121_{timestamp}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging for this evaluation
            setup_logging(output_dir)
            logging.info(f"Starting ModelConfig evaluation for {args.task} task on {args.test_split} split with densenet121 model")
            
            # Log implementation details
            logging.info(f"Implementation details:")
            logging.info(f"  Architecture: densenet121")
            logging.info(f"  Epochs: {config.get('epochs', 50)}")
            logging.info(f"  Optimizer: {config.get('optimizer', {}).get('name', 'AdamW')}")
            logging.info(f"  Learning Rate: {config.get('optimizer', {}).get('learning_rate', 0.0002)}")
            logging.info(f"  Batch Size: {config.get('batch_size', 32)}")
            logging.info(f"  Positive Class Weight: {config.get('pos_weight', 0.5)}")
            
            try:
                # Setup device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logging.info(f"Using device: {device}")
                
                # Check multiple locations for existing models
                model_path_options = [
                    paths["MODELS_DIR"] / f"best_model_{args.task}_densenet121.pt",
                    paths["EXPERIMENT_MODELS_DIR"] / f"best_model_{args.task}_densenet121.pt",
                    paths["EXPERIMENT_MODELS_DIR"] / f"densenet121_{args.task}_ModelConfig_best.pt"
                ]
                
                model_path = None
                for path_option in model_path_options:
                    if path_option.exists():
                        model_path = path_option
                        logging.info(f"Found existing model at: {model_path}")
                        break
                        
                if model_path is None:
                    # Default to the new path if no existing model is found
                    model_path = paths["EXPERIMENT_MODELS_DIR"] / f"densenet121_{args.task}_ModelConfig_best.pt"
                    logging.warning(f"No existing model found. Will attempt to train and save to: {model_path}")
                logging.info(f"Looking for model at: {model_path}")
                
                # Load model
                model = load_model(model_path, device, 'densenet121')
                
                # Create validation data directory
                validation_dir = output_dir / "validation_thresholds"
                validation_dir.mkdir(exist_ok=True)
                
                # For ModelConfig, use actual model paths from the evaluate.py results
                # This ensures identical thresholds for both approaches
                if args.task == 'inflammation':
                    threshold_path = Path('/mnt/data/dliebel/2024_dliebel/results/evaluations/inflammation_scanner1_densenet121_20250317_232234/validation_thresholds/thresholds.json')
                else:  # tissue task
                    threshold_path = Path('/mnt/data/dliebel/2024_dliebel/results/evaluations/tissue_scanner1_densenet121_20250317_225530/validation_thresholds/thresholds.json')
                
                # Load the validation thresholds
                if threshold_path.exists():
                    try:
                        with open(threshold_path, 'r') as f:
                            validation_thresholds = json.load(f)
                        logging.info(f"Loaded validation thresholds from {threshold_path}")
                        logging.info("Using pre-computed thresholds ensures proper validation/test separation")
                    except Exception as e:
                        logging.error(f"Error loading threshold file: {str(e)}")
                        validation_thresholds = None
                else:
                    logging.warning(f"Threshold file not found: {threshold_path}")
                    validation_thresholds = None
                
                # If unable to load pre-computed thresholds, calculate them
                if validation_thresholds is None:
                    logging.info("Calculating validation thresholds...")
                    validation_thresholds = metrics_utils.calculate_validation_thresholds(
                        model_path=model_path,
                        model=model,
                        task=args.task,
                        output_dir=validation_dir
                    )
                    
                    # Save thresholds for future use
                    threshold_save_path = validation_dir / "thresholds.json"
                    with open(threshold_save_path, 'w') as f:
                        json.dump(validation_thresholds, f, indent=2)
                    logging.info(f"Saved calculated validation thresholds to {threshold_save_path}")
                
                # Create test dataset with the same transforms used in evaluate.py
                test_dataset = HistologyDataset(
                    split=args.test_split,
                    transform=get_transforms(is_training=False),  # Use the same transforms as evaluate.py
                    task=args.task,
                    paths=paths
                )
                
                # Create test dataloader with identical parameters to evaluate.py
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=32,  # Same batch size as evaluate.py
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Run evaluation using evaluate_model from evaluate.py
                metrics = evaluate.evaluate_model(
                    model=model,
                    data_loader=test_loader,
                    task=args.task,
                    split=args.test_split,
                    device=device,
                    output_dir=output_dir,
                    validation_thresholds=validation_thresholds
                )
                
                # Add implementation details to metrics
                metrics["implementation_details"] = {
                    "epochs": config.get('epochs', 50),
                    "optimizer": config.get('optimizer', {}).get('name', 'AdamW'),
                    "learning_rate": config.get('optimizer', {}).get('learning_rate', 0.0002),
                    "batch_size": config.get('batch_size', 32),
                    "pos_class_weight": config.get('pos_weight', 0.5)
                }
                
                # Save metrics using the same approach as evaluate.py
                metrics_utils.save_metrics(metrics, output_dir)
                
                # Create ROC curves
                if 'predictions_df' in metrics:
                    df = metrics['predictions_df']
                    # Use metrics_utils.plot_roc_curves just like in evaluate.py
                    metrics_utils.plot_roc_curves(
                        df,
                        args.task,
                        output_dir,
                        metrics.get('optimal_aggregation', None)
                    )
                else:
                    logging.warning("Could not find predictions DataFrame in metrics. Skipping ROC curve generation.")
                
                # Create precision-recall curves
                if 'predictions_df' in metrics:
                    try:
                        evaluate.plot_precision_recall_curve(metrics['predictions_df'], args.task, output_dir)
                        logging.info("Precision-recall curves saved successfully")
                    except Exception as e:
                        logging.warning(f"Failed to create precision-recall curves: {str(e)}")
                else:
                    logging.warning("Could not find predictions DataFrame in metrics. Skipping precision-recall curve generation.")
                    
                logging.info("ModelConfig evaluation completed successfully!")
                
                # Return from the function early since we've completed the evaluation
                return
                
            except Exception as e:
                logging.error(f"ModelConfig evaluation failed: {str(e)}")
                raise
    
    roc_tile_all = {}
    roc_agg_all = {}

    for variant_name, aug_params in augmentation_variants.items():
        logging.info(f"--- {args.task.capitalize()} Variant: {variant_name} ---")

        if aug_params["only_normalization"]:
            train_transform = [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        elif aug_params.get("use_model_config", False):
            # Get base transforms from model config
            train_transform = get_transforms(is_training=True)
            
            # Add additional augmentations if specified
            if aug_params["use_medmnist_aug"] and MEDMNISTC_AVAILABLE:
                if isinstance(train_transform, transforms.Compose):
                    # Convert to list for modification
                    transform_list = list(train_transform.transforms)
                    # Add MedMNIST augmentation at the start
                    medmnist_aug = AugMedMNISTC(CORRUPTIONS_DS["pathmnist"])
                    transform_list.insert(0, MedMNISTAugTransform(medmnist_aug))
                    # Rebuild compose
                    train_transform = transforms.Compose(transform_list)
            
            if aug_params["use_stain_color_jitter"]:
                if isinstance(train_transform, transforms.Compose):
                    transform_list = list(train_transform.transforms)
                    # Find position to insert StainColorJitter (after ToTensor but before Normalize)
                    to_tensor_pos = -1
                    normalize_pos = -1
                    for i, t in enumerate(transform_list):
                        if isinstance(t, transforms.ToTensor):
                            to_tensor_pos = i
                        if isinstance(t, transforms.Normalize):
                            normalize_pos = i
                            break
                    
                    if to_tensor_pos >= 0 and normalize_pos > to_tensor_pos:
                        transform_list.insert(normalize_pos, StainColorJitter(sigma=0.05))
                        train_transform = transforms.Compose(transform_list)
        else:
            # Your existing base transforms code for non-ModelConfig variants
            base_transforms = [
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor()
            ]
            
            if aug_params["use_medmnist_aug"] and MEDMNISTC_AVAILABLE:
                medmnist_aug = AugMedMNISTC(CORRUPTIONS_DS["pathmnist"])
                base_transforms.insert(0, MedMNISTAugTransform(medmnist_aug))
            elif aug_params["use_medmnist_aug"]:
                logging.warning("Medmnist augmentation unavailable; skipping")
            
            if aug_params["use_stain_color_jitter"]:
                base_transforms.append(StainColorJitter(sigma=0.05))
            
            base_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            train_transform = base_transforms

        if not isinstance(train_transform, transforms.Compose):
            logging.info(f"Converting train_transform to Compose for {variant_name}. Original type: {type(train_transform)}")
            train_transform = transforms.Compose(train_transform)

        val_transform = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        val_transform = transforms.Compose(val_transform)

        logging.info(f"Train transform type for {variant_name}: {type(train_transform)}")
        logging.info(f"Val transform type for {variant_name}: {type(val_transform)}")
        
        train_dataset = HistologyDataset(split='train', transform=train_transform, task=args.task, paths=paths)
        val_dataset = HistologyDataset(split='val', transform=val_transform, task=args.task, paths=paths)
        test_dataset = HistologyDataset(split=args.test_split, transform=val_transform, task=args.task, paths=paths)
        
        num_workers = min(8, 4 * num_gpus) if num_gpus > 0 else 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'] * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'] * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        from src.models.architectures import HistologyClassifier
        model = HistologyClassifier(
            model_name='densenet121',
            num_classes=1 if args.task == 'inflammation' else 2,
            dropout_rate=config['dropout_rate']
        )
        if num_gpus > 1:
            model = training_utils.handle_multi_gpu_model(model)
        model = model.to(device)

        # Define criterion based on task
        if args.task == 'inflammation':
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([config['pos_weight']]).to(device)
            )
        elif args.task == 'tissue':
            class_weights = torch.tensor([1.0, config['pos_weight']]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # Configure optimizer and scaler
        training_components = training_utils.configure_training_components(
            model=model,
            learning_rate=config['optimizer']['learning_rate'],
            weight_decay=config['optimizer']['weight_decay'],
            pos_weight=config['pos_weight'],
            optimizer_name=config['optimizer']['name'],
            device=device
        )
        optimizer = training_components['optimizer']
        scaler = training_components['scaler']

        if config['scheduler']['enabled'] and config['scheduler']['name'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler']['T_max'])
        else:
            scheduler = None

        best_val_loss = float('inf')
        patience = config.get('early_stopping', {}).get('patience', 10)
        epochs_no_improve = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        for epoch in range(config['epochs']):
            if args.task == 'tissue':
                # Custom train_epoch for tissue task
                def tissue_train_epoch(model, dataloader, optimizer, criterion, scaler, device, use_amp=True):
                    model.train()
                    total_loss = 0.0
                    all_preds = []
                    all_labels = []
                    all_raw_preds = []
                    
                    for inputs, labels, _ in dataloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device).long()  # Ensure labels are long for CrossEntropyLoss
                        
                        optimizer.zero_grad()
                        
                        if use_amp:
                            with torch.amp.autocast(device_type='cuda'):
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                        
                        # Store predictions and labels
                        preds = torch.argmax(outputs, dim=1)
                        probs = torch.softmax(outputs, dim=1)[:, 1]
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_raw_preds.extend(probs.detach().cpu().numpy())
                        
                        total_loss += loss.item()
                    
                    metrics = metrics_utils.calculate_basic_metrics(all_labels, all_preds, all_raw_preds)
                    metrics['loss'] = total_loss / len(dataloader)
                    return metrics

                # Custom validate for tissue task
                def tissue_validate(model, dataloader, criterion, device):
                    model.eval()
                    total_loss = 0.0
                    all_preds = []
                    all_labels = []
                    all_raw_preds = []
                    
                    with torch.no_grad():
                        for inputs, labels, _ in dataloader:
                            inputs = inputs.to(device)
                            labels = labels.to(device).long()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            
                            preds = torch.argmax(outputs, dim=1)
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                            all_raw_preds.extend(probs.cpu().numpy())
                            
                            total_loss += loss.item()
                    
                    metrics = metrics_utils.calculate_basic_metrics(all_labels, all_preds, all_raw_preds)
                    metrics['loss'] = total_loss / len(dataloader)
                    return metrics
                    

                train_metrics = tissue_train_epoch(
                    model, train_loader, optimizer, criterion, scaler, device, use_amp=True
                )
                val_metrics = tissue_validate(model, val_loader, criterion, device)
            else:
                # Use standard training_utils functions for inflammation
                train_metrics = training_utils.train_epoch(
                    model, train_loader, optimizer, criterion, scaler, device, use_amp=True
                )
                val_metrics = training_utils.validate(model, val_loader, criterion, device)

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])

            logging.info(f"Epoch {epoch+1}/{config['epochs']} ({variant_name}): Train Loss={train_metrics['loss']:.4f}, Val Loss={val_metrics['loss']:.4f}")

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_no_improve = 0
                model_path = paths["EXPERIMENT_MODELS_DIR"] / f"densenet121_{args.task}_{variant_name}_best.pt"
                training_utils.save_checkpoint(
                    model, optimizer, epoch, val_metrics, config, str(model_path), seed=42
                )
                logging.info(f"Saved best model at {model_path}")
            else:
                epochs_no_improve += 1
                if config.get('early_stopping', {}).get('enabled', False) and epochs_no_improve >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs for {variant_name}")
                    break

            if scheduler:
                scheduler.step()

        # Load best model using the same approach as evaluate.py
        loaded_data = load_model(model_path, device, architecture='densenet121')
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            model = loaded_data['model']
        else:
            model = loaded_data
        if num_gpus > 1:
            model = training_utils.handle_multi_gpu_model(model)
        model = model.to(device)

        # Load pre-computed thresholds based on task
        if args.task == 'inflammation':
            threshold_path = Path('/mnt/data/dliebel/2024_dliebel/results/evaluations/inflammation_scanner1_densenet121_20250317_232234/validation_thresholds/thresholds.json')
        else:  # tissue task
            threshold_path = Path('/mnt/data/dliebel/2024_dliebel/results/evaluations/tissue_scanner1_densenet121_20250317_225530/validation_thresholds/thresholds.json')

        # Load the pre-computed thresholds
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    validation_thresholds = json.load(f)
                logging.info(f"Loaded pre-computed validation thresholds from {threshold_path}")
            except Exception as e:
                logging.error(f"Error loading threshold file: {str(e)}")
                # Fall back to default thresholds
                validation_thresholds = {
                    'tile': {'threshold': 0.616, 'sensitivity': 0.8, 'specificity': 0.8, 'f1': 0.8},
                    'slide' if args.task == 'inflammation' else 'particle': {
                        'threshold': 0.850 if args.task == 'inflammation' else 0.568,
                        'sensitivity': 0.8, 
                        'specificity': 0.8, 
                        'f1': 0.8
                    }
                }
                logging.warning("Using default thresholds due to loading error.")
        else:
            logging.warning(f"Threshold file not found at {threshold_path}. Using default thresholds.")
            # Use default thresholds
            validation_thresholds = {
                'tile': {'threshold': 0.616, 'sensitivity': 0.8, 'specificity': 0.8, 'f1': 0.8},
                'slide' if args.task == 'inflammation' else 'particle': {
                    'threshold': 0.850 if args.task == 'inflammation' else 0.568,
                    'sensitivity': 0.8, 
                    'specificity': 0.8, 
                    'f1': 0.8
                }
            }

        # Evaluate model using the single evaluation method from evaluate.py
        metrics = evaluate.evaluate_model(
            model=model,
            data_loader=test_loader,
            task=args.task,
            split=args.test_split,
            device=device,
            output_dir=paths["EXPERIMENTS_DIR"],
            validation_thresholds=validation_thresholds
        )
        
        # Extract predictions dataframe for plotting
        if 'predictions_df' in metrics:
            df = metrics['predictions_df']
            
            # Get true labels and prediction probabilities for tile-level ROC curve
            true_labels = df['label'].values
            pred_probs = df['prob'].values if 'prob' in df.columns else df['pred_prob'].values
            
            # Compute ROC curves for tile-level predictions
            tile_fpr, tile_tpr, _ = roc_curve(true_labels, pred_probs)
            
            # Use the AUC values directly from metrics instead of recalculating
            tile_auc = metrics.get('tile_auc', metrics.get('tile_auroc', 0))
            roc_tile_all[variant_name] = (tile_fpr, tile_tpr, tile_auc)
            
            # Get slide/particle-level predictions directly from metrics
            if args.task == 'inflammation':
                # Extract slide-level metrics
                agg_level = 'slide'
                agg_auc = metrics.get('slide_auc', metrics.get('slide_auroc', 0))
                
                # Compute slide-level ROC curve from aggregated predictions in metrics
                if 'slide_df' in metrics:
                    slide_df = metrics['slide_df']
                    agg_true = slide_df['label'].values
                    agg_probs = slide_df['prob'].values if 'prob' in slide_df.columns else slide_df['pred_prob'].values
                    agg_fpr, agg_tpr, _ = roc_curve(agg_true, agg_probs)
                else:
                    # Fallback to aggregating from the dataframe if slide_df not available
                    groups = df['slide_name'].values
                    unique_groups = np.unique(groups)
                    agg_true, agg_probs = [], []
                    for g in unique_groups:
                        idx = np.where(groups == g)[0]
                        agg_true.append(mode(true_labels[idx])[0])
                        # Use aggregation method from evaluate.py
                        agg_probs.append(np.mean(pred_probs[idx]))  # inflammation uses mean
                    agg_true, agg_probs = np.array(agg_true), np.array(agg_probs)
                    agg_fpr, agg_tpr, _ = roc_curve(agg_true, agg_probs)
            else:  # tissue task
                # Extract particle-level metrics
                agg_level = 'particle'
                agg_auc = metrics.get('particle_auc', metrics.get('particle_auroc', 0))
                
                # Compute particle-level ROC curve from aggregated predictions in metrics
                if 'particle_df' in metrics:
                    particle_df = metrics['particle_df']
                    agg_true = particle_df['label'].values
                    agg_probs = particle_df['prob'].values if 'prob' in particle_df.columns else particle_df['pred_prob'].values
                    agg_fpr, agg_tpr, _ = roc_curve(agg_true, agg_probs)
                else:
                    # Fallback to aggregating from the dataframe if particle_df not available
                    groups = df['particle_id'].values if 'particle_id' in df.columns else df['slide_name'].values
                    unique_groups = np.unique(groups)
                    agg_true, agg_probs = [], []
                    for g in unique_groups:
                        idx = np.where(groups == g)[0]
                        agg_true.append(mode(true_labels[idx])[0])
                        # Use aggregation method from evaluate.py
                        agg_probs.append(np.median(pred_probs[idx]))  # tissue uses median
                    agg_true, agg_probs = np.array(agg_true), np.array(agg_probs)
                    agg_fpr, agg_tpr, _ = roc_curve(agg_true, agg_probs)
            
            roc_agg_all[variant_name] = (agg_fpr, agg_tpr, agg_auc)
        else:
            # Fallback if predictions_df is not available in metrics
            logging.warning(f"predictions_df not found in metrics for {variant_name}. Using default ROC curves.")
            # Create empty placeholder ROC curves
            roc_tile_all[variant_name] = (np.array([0, 1]), np.array([0, 1]), 0.5)
            roc_agg_all[variant_name] = (np.array([0, 1]), np.array([0, 1]), 0.5)
        
        scanner_name = "Scanner1" if args.test_split == 'test' else "Scanner2"
        level = "Slide" if args.task == 'inflammation' else "Particle"
        
        # Define aggregation method name based on task
        agg_method_name = "Mean" if args.task == 'inflammation' else "Median"

        # Individual plots for this variant
        plot_roc_curves(
            {f"Tile-Level ({variant_name})": roc_tile_all[variant_name]},
            title=f"{args.task.capitalize()} - Tile-Level ROC ({scanner_name}) - {variant_name}",
            save_path=str(paths["FIGURES_DIR"] / f"roc_{args.task}_tile_{args.test_split}_{variant_name}.png")
        )
        plot_roc_curves(
            {f"{level}-Level ({agg_method_name})": roc_agg_all[variant_name]},
            title=f"{args.task.capitalize()} - {level}-Level ROC ({scanner_name}) - {variant_name}",
            save_path=str(paths["FIGURES_DIR"] / f"roc_{args.task}_{level.lower()}_{args.test_split}_{variant_name}.png")
        )
        
        # Save metrics to disk - ensure we use consistent metrics for ROC curves and text file results
        metrics_utils.save_metrics(metrics, paths["EXPERIMENTS_DIR"])
        with open(paths["EXPERIMENTS_DIR"] / f"history_{variant_name}.json", 'w') as f:
            json.dump(history, f, indent=2)
            
        # Extract metrics for results file - only use metrics from evaluate.evaluate_model
        tile_metrics = {k: v for k, v in metrics.items() if k.startswith('tile_') and isinstance(v, (int, float, np.floating))}
        
        # Use correct prefix based on task (slide for inflammation, particle for tissue)
        if args.task == 'inflammation':
            agg_prefix = 'slide_'
            agg_metrics = {k: v for k, v in metrics.items() if k.startswith(agg_prefix) and isinstance(v, (int, float, np.floating))}
        else:
            agg_prefix = 'particle_'
            agg_metrics = {k: v for k, v in metrics.items() if k.startswith(agg_prefix) and isinstance(v, (int, float, np.floating))}
        
        # Write results to text file - ensure consistency with evaluate.py
        with open(paths["EXPERIMENTS_DIR"] / f"results_{args.task}_{variant_name}_{args.test_split}.txt", 'w') as f:
            f.write(f"{args.task.upper()} RESULTS - Variant: {variant_name} - Scanner: {args.test_split}\n")
            f.write("========================================\n\n")
            f.write(f"TEST DATASET - TILE LEVEL METRICS\n")
            f.write("---------------------------------------------\n")
            for k, v in tile_metrics.items():
                f.write(f"{k.replace('tile_', '').capitalize()}: {v:.3f}\n")
            
            level_name = "SLIDE" if args.task == 'inflammation' else "PARTICLE"
            f.write(f"\nTEST DATASET - {level_name}-LEVEL METRICS\n")
            f.write("---------------------------------------------\n")
            for k, v in agg_metrics.items():
                metric_name = k.replace(agg_prefix, '')
                f.write(f"{metric_name.capitalize()}: {v:.3f}\n")

    # This block is OUTSIDE the variant loop
    if not args.variant and len(augmentation_variants) > 1:
        # Comparison plots for all variants
        scanner_name = "Scanner1" if args.test_split == 'test' else "Scanner2"
        level = "Slide" if args.task == 'inflammation' else "Particle"
        
        # Tile-level comparison
        plot_roc_curves(
            {variant_name: roc_data for variant_name, roc_data in roc_tile_all.items()},
            title=f"{args.task.capitalize()} - Tile-Level ROC Comparison ({scanner_name}) - All Variants",
            save_path=str(paths["FIGURES_DIR"] / f"roc_{args.task}_tile_{args.test_split}_all_variants.png")
        )
        logging.info(f"Tile-level ROC comparison plot saved to {paths['FIGURES_DIR'] / f'roc_{args.task}_tile_{args.test_split}_all_variants.png'}")
        
        # Aggregated level comparison
        plot_roc_curves(
            {variant_name: roc_data for variant_name, roc_data in roc_agg_all.items()},
            title=f"{args.task.capitalize()} - {level}-Level ROC Comparison ({scanner_name}) - All Variants",
            save_path=str(paths["FIGURES_DIR"] / f"roc_{args.task}_{level.lower()}_{args.test_split}_all_variants.png")
        )
        logging.info(f"Aggregated-level ROC comparison plot saved to {paths['FIGURES_DIR'] / f'roc_{args.task}_{level.lower()}_{args.test_split}_all_variants.png'}")

def parse_args():
    """Parse command-line arguments with 'all' option for task and test_split."""
    parser = argparse.ArgumentParser(description="Run experiments with variants, hardcoded thresholds, and aggregation")
    parser.add_argument("--task", choices=["inflammation", "tissue", "all"], default="all",
                        help="Task to run or 'all' for both (default: all)")
    parser.add_argument("--test_split", choices=["test", "test_scanner2", "all"], default="all",
                        help="Test split to use or 'all' for both (default: all)")
    parser.add_argument("--variant", choices=["Medmnist", "ColorJitter", "ModelConfig", "Medmnist_ColorJitter", "All", "NormalizationOnly"],
                        default=None, help="Optional: Specific augmentation variant to test (default: all)")
    parser = add_path_args(parser)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    paths = get_project_paths()
    paths["RESULTS_DIR"].mkdir(parents=True, exist_ok=True)
    paths["EXPERIMENTS_DIR"] = paths["RESULTS_DIR"] / "experiments"
    paths["EXPERIMENTS_DIR"].mkdir(parents=True, exist_ok=True)
    paths["MODELS_DIR"].mkdir(parents=True, exist_ok=True)
    paths["FIGURES_DIR"] = paths.get("FIGURES_DIR", paths["EXPERIMENTS_DIR"] / "figures")
    paths["FIGURES_DIR"].mkdir(parents=True, exist_ok=True)
    paths["EXPERIMENT_MODELS_DIR"] = paths["EXPERIMENTS_DIR"] / "models"
    paths["EXPERIMENT_MODELS_DIR"].mkdir(parents=True, exist_ok=True)
    logging.info(f"Models will be saved to: {paths['EXPERIMENT_MODELS_DIR']}")
    

    all_tasks = ["inflammation", "tissue"] if args.task == "all" else [args.task]
    all_test_splits = ["test", "test_scanner2"] if args.test_split == "all" else [args.test_split]

    for task in all_tasks:
        for test_split in all_test_splits:
            logging.info(f"Starting experiment: Task={task}, Test Split={test_split}")
            args.task = task
            args.test_split = test_split
            train_and_evaluate(args, paths)
            logging.info(f"Completed experiment: Task={task}, Test Split={test_split}")