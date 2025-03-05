import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBomb warning
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple
import uuid
import re
from shapely.geometry import Polygon, Point
import sys
import cv2
from skimage import measure
import threading
from queue import Queue
import traceback

class AnnotationType(Enum):
    TISSUE = "tissue"
    INFLAMMATION = "inflammation"

class WSIListFrame(ttk.Frame):
    def __init__(self, parent, callback, downsample=16, annotation_folder=None):
        super().__init__(parent)
        self.callback = callback
        self.DOWNSAMPLE = downsample
        self.parent = parent
        self.annotation_folder = annotation_folder

        # Create necessary subdirectories
        (self.annotation_folder / "clusters").mkdir(parents=True, exist_ok=True)
        
        # Add thumbnail cache
        self.thumbnail_cache = {}
        self.MAX_THUMBNAIL_CACHE = 50
        
        # Add queue for lazy loading
        self.load_queue = Queue()
        self.current_loads = set()
        
        # Initialize dictionaries for storing references
        self.name_labels = {}
        self.thumbnail_refs = {}
        
        # Title
        title = ttk.Label(self, text="Available slides")
        title.pack(pady=5)
        
        # Create canvas and scrollbar for WSI list
        self.canvas = tk.Canvas(self, width=180)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Create inner frame for WSIs
        self.inner_frame = ttk.Frame(self.canvas)
        self.inner_frame_id = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        
        # Configure scrolling
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind events for proper scrolling
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Trackpad and mouse wheel scrolling bindings
        self.canvas.bind("<MouseWheel>", self._handle_scroll)
        self.canvas.bind("<Button-4>", self._handle_scroll)
        self.canvas.bind("<Button-5>", self._handle_scroll)
        self.canvas.bind_all("<MouseWheel>", self._handle_scroll)
        
        # Bind enter/leave for scroll events
        self.canvas.bind("<Enter>", self._bind_scroll)
        self.canvas.bind("<Leave>", self._unbind_scroll)

    def _bind_scroll(self, event=None):
        """Bind all scroll events when mouse enters the canvas"""
        self.canvas.bind_all("<MouseWheel>", self._handle_scroll)
        self.canvas.bind_all("<Button-4>", self._handle_scroll)
        self.canvas.bind_all("<Button-5>", self._handle_scroll)
    
    def _unbind_scroll(self, event=None):
        """Unbind scroll events when mouse leaves the canvas"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _handle_scroll(self, event):
        """Handle all types of scrolling events"""
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            # For Windows and macOS trackpad
            delta = event.delta
            
            # Convert macOS large values
            if abs(delta) > 100:
                delta = int(delta/120)
                
            self.canvas.yview_scroll(int(-0.5 * delta), "units")
        
        # Prevent event propagation
        return "break"
    
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame to match"""
        self.canvas.itemconfig(self.inner_frame_id, width=event.width)

    def add_wsi(self, path: Path, name: str):
        """Add a WSI preview to the list with lazy loading"""
        # Create frame for this WSI
        wsi_frame = ttk.Frame(self.inner_frame)
        wsi_frame.pack(fill="x", padx=5, pady=2)
        
        try:
            # Get original image size without loading full image
            with Image.open(path) as img:
                original_size = img.size
            
            # Create canvas for thumbnail placeholder
            thumb_canvas = tk.Canvas(
                wsi_frame,
                width=180,
                height=120,
                highlightthickness=0
            )
            thumb_canvas.pack()
            
            # Add loading placeholder
            thumb_canvas.create_text(
                90, 60,
                text="Loading...",
                fill="gray"
            )
            
            # Add WSI name with annotation and cluster counts
            annotation_count = self.get_annotation_count(name)
            cluster_count = self.get_cluster_count(name)
            name_text = f"{name} ({annotation_count} Annotations, {cluster_count} Clusters)"
            name_label = ttk.Label(wsi_frame, text=name_text)
            name_label.pack()
    
            
            # Store references with original size
            self.thumbnail_refs[name] = {
                'canvas': thumb_canvas,
                'frame': wsi_frame,
                'path': path,
                'original_size': original_size
            }
            
            # Store references
            self.name_labels[name] = name_label
            
            # Make both labels clickable
            thumb_canvas.bind("<Button-1>", lambda e, p=path: self.callback(p))
            name_label.bind("<Button-1>", lambda e, p=path: self.callback(p))
            
            # Queue thumbnail for loading
            self.queue_thumbnail_load(name, path, thumb_canvas)
            
        except Exception as e:
            logging.error(f"Error adding slide preview for {name}: {e}")
            if wsi_frame:
                wsi_frame.destroy()

    def queue_thumbnail_load(self, name: str, path: Path, canvas: tk.Canvas):
        """Queue a thumbnail for background loading"""
        if name not in self.current_loads:
            self.current_loads.add(name)
            threading.Thread(
                target=self.load_thumbnail,
                args=(name, path, canvas),
                daemon=True
            ).start()

    def load_thumbnail(self, name: str, path: Path, canvas: tk.Canvas):
        """Load thumbnail in background"""
        try:
            # Check cache first
            if path in self.thumbnail_cache:
                photo = self.thumbnail_cache[path]
                self.parent.after_idle(  # Use parent instead of root
                    lambda: self.update_thumbnail(name, photo, canvas)
                )
                return
            
            # Load and resize thumbnail
            img = Image.open(path)
            img.thumbnail((180, 540))
            
            # Convert to PhotoImage in main thread
            self.parent.after_idle(  # Use parent instead of root
                lambda: self.create_and_cache_thumbnail(name, img, path, canvas)
            )
            
        except Exception as e:
            logging.error(f"Error loading thumbnail for {name}: {e}")
            self.current_loads.discard(name)
            
    def create_and_cache_thumbnail(self, name: str, img, path: Path, canvas: tk.Canvas):
        """Create PhotoImage and cache in main thread"""
        try:
            # Create PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Manage cache size
            if len(self.thumbnail_cache) >= self.MAX_THUMBNAIL_CACHE:
                oldest = next(iter(self.thumbnail_cache))
                del self.thumbnail_cache[oldest]
            
            # Cache thumbnail
            self.thumbnail_cache[path] = photo
            
            # Update display
            self.update_thumbnail(name, photo, canvas)
            
        except Exception as e:
            logging.error(f"Error creating thumbnail for {name}: {e}")
        finally:
            self.current_loads.discard(name)

    def update_thumbnail(self, name: str, photo: ImageTk.PhotoImage, canvas: tk.Canvas):
        """Update canvas with loaded thumbnail"""
        try:
            # Clear loading placeholder
            canvas.delete("all")
            
            # Update canvas size and add image
            canvas.config(width=photo.width(), height=photo.height())
            canvas.create_image(0, 0, image=photo, anchor="nw", tags="thumbnail")
            
            # Store reference
            if name in self.thumbnail_refs:
                self.thumbnail_refs[name]['photo'] = photo
                
        except Exception as e:
            logging.error(f"Error updating thumbnail for {name}: {e}")

    def update_viewport(self, wsi_name: str, view_region: tuple):
        """Update viewport overlay for the given WSI"""
        if wsi_name not in self.thumbnail_refs:
            return
                
        try:
            thumb_info = self.thumbnail_refs[wsi_name]
            thumb_canvas = thumb_info['canvas']
            
            if 'photo' not in thumb_info:
                return
                
            thumb_width = thumb_info['photo'].width()
            thumb_height = thumb_info['photo'].height()
            
            # Clear previous viewport
            thumb_canvas.delete("viewport")
            
            # Unpack view region (these are now fractions between 0 and 1)
            x_scroll, y_scroll, view_width_fraction, view_height_fraction = view_region
            
            # Calculate viewport coordinates in thumbnail space
            x1 = x_scroll * thumb_width
            y1 = y_scroll * thumb_height
            x2 = x1 + (view_width_fraction * thumb_width)
            y2 = y1 + (view_height_fraction * thumb_height)
            
            # Ensure viewport stays within bounds
            x1 = max(0, min(x1, thumb_width))
            y1 = max(0, min(y1, thumb_height))
            x2 = min(thumb_width, max(x1, x2))
            y2 = min(thumb_height, max(y1, y2))
            
            # Draw viewport rectangle
            thumb_canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red",
                width=1,
                tags="viewport"
            )
            
        except Exception as e:
            logging.error(f"Error in update_viewport: {str(e)}")

    def get_annotation_count(self, wsi_name: str) -> int:
        """Get number of annotations for a WSI"""
        try:
            annotation_path = self.annotation_folder / f"{wsi_name}_annotations.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)
                return len(annotations)
            return 0
        except Exception as e:
            logging.error(f"Error getting annotation count for {wsi_name}: {e}")
            return 0

    def update_annotation_count(self, wsi_name: str):
        """Update the annotation and cluster count display for a WSI"""
        if wsi_name in self.name_labels:
            annotation_count = self.get_annotation_count(wsi_name)
            cluster_count = self.get_cluster_count(wsi_name)
            name_text = f"{wsi_name} ({annotation_count} Annotations, {cluster_count} Clusters)"
            self.name_labels[wsi_name].config(text=name_text)

    def update_cluster_display(self, wsi_name: str):
        """Update display after cluster changes"""
        self.update_annotation_count(wsi_name)

    def get_cluster_count(self, wsi_name: str) -> int:
        """Get number of clusters for a WSI"""
        try:
            cluster_path = self.annotation_folder / "clusters" / f"{wsi_name}_clusters.json"
            if cluster_path.exists():
                with open(cluster_path, 'r') as f:
                    clusters = json.load(f)
                    return len(clusters)
            return 0
        except Exception as e:
            logging.error(f"Error getting cluster count for {wsi_name}: {e}")
            return 0
    
    def clear(self):
        """Clear all WSI previews"""
        self.name_labels.clear()
        self.thumbnail_refs.clear()
        for widget in self.inner_frame.winfo_children():
            widget.destroy()

class AnnotationTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WSI Annotation Tool")

        # Set window to full screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        
        # Select annotation folder at startup
        self.select_annotation_folder()
        
        # Load configuration if it exists - moved after select_annotation_folder
        self.load_config()
        
        # Remove model selection at startup - will be done when classifying
        # self.select_model_paths()  # <-- Remove this line
        
        # Initialize model paths to None
        self.tissue_model_path = None
        self.inflammation_model_path = None
        
        # Add memory management
        self.image_cache = {}
        self.max_cache_size = 10  # Number of WSIs to keep in memory
        
        # Add auto-backup configuration
        self.backup_interval = 300000  # 5 minutes in milliseconds
        
        # Constants
        self.DEFAULT_WIDTH = 1200
        self.DEFAULT_HEIGHT = 800
        self.DOWNSAMPLE = 16  # Fixed downsample factor for PNG files
        
        # Annotation options with vibrant colors for tissue types only
        self.ANNOTATION_OPTIONS = {
            'tissue': {
                'corpus': [255, 50, 50],       # Bright red
                'antrum': [50, 50, 255],       # Bright blue
                'intermediate': [50, 255, 50],  # Bright green
                'other': [189, 0, 255]         # Bright purple
            },
            'inflammation_status': {  # Changed from 'inflammation' for consistency
                'inflamed': None,      # No separate colors for inflammation
                'noninflamed': None,
                'unclear': None,
                'other': None
            }
        }

        # Default inflammation status for quick annotations
        self.default_inflammation_status = 'other'
        
        # State variables
        self.current_image = None
        self.current_points = []
        self.annotations = []
        self.drawing = False
        self.current_wsi_name = None

        # Cluster-related state variables
        self.cluster_start = None  # First corner of cluster
        self.clusters = []  # List to store clusters
        self.show_clusters = True  # For toggling cluster visibility
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.original_width = None
        self.original_height = None
        
        # Setup UI components
        self.setup_ui()
        self.setup_bindings()
        self.setup_keyboard_shortcuts()
        
        # Initialize auto-backup
        self.setup_auto_backup()

    def load_config(self):
        """Load configuration from file"""
        if not hasattr(self, 'annotation_folder'):
            logging.error("Annotation folder not set before loading config")
            self.config = {}
            return
        
        self.config_file = self.annotation_folder / "annotation_config.json"
        self.config = {}
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logging.error(f"Error loading config: {e}")
                self.config = {}
        
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def select_model_paths(self):
        """Allow user to select paths and architectures for tissue and inflammation models"""
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model Paths and Architectures")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        window_width = 700
        window_height = 350
        position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
        position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
        dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Get last used paths from config
        tissue_path = self.config.get('tissue_model_path', '')
        inflammation_path = self.config.get('inflammation_model_path', '')
        tissue_arch = self.config.get('tissue_model_architecture', 'resnet18')
        inflammation_arch = self.config.get('inflammation_model_architecture', 'resnet18')
        
        # Available architectures
        available_architectures = [
            'gigapath',
            'resnet18',
            'swin_v2_b',
            'convnext_large',
            'densenet121',
            'densenet169'
        ]
        
        # Tissue model path and architecture
        tissue_frame = ttk.LabelFrame(dialog, text="Tissue Model Configuration")
        tissue_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Path selection
        path_frame = ttk.Frame(tissue_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(path_frame, text="Model Path:").pack(side=tk.LEFT, padx=5)
        tissue_var = tk.StringVar(value=tissue_path)
        tissue_entry = ttk.Entry(path_frame, textvariable=tissue_var, width=60)
        tissue_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_tissue():
            path = filedialog.askopenfilename(
                title="Select Tissue Model",
                filetypes=[("PyTorch Models", "*.pt *.pth"), ("All Files", "*.*")],
                initialdir=str(Path(tissue_var.get()).parent) if tissue_var.get() else "/"
            )
            if path:
                tissue_var.set(path)
        
        ttk.Button(path_frame, text="Browse...", command=browse_tissue).pack(side=tk.RIGHT, padx=5)
        
        # Architecture selection
        arch_frame = ttk.Frame(tissue_frame)
        arch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(arch_frame, text="Architecture:").pack(side=tk.LEFT, padx=5)
        tissue_arch_var = tk.StringVar(value=tissue_arch)
        tissue_arch_menu = ttk.Combobox(arch_frame, textvariable=tissue_arch_var, values=available_architectures)
        tissue_arch_menu.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add architecture description
        tissue_arch_desc = ttk.Label(tissue_frame, text="Model architecture type (should match your trained model)")
        tissue_arch_desc.pack(fill=tk.X, padx=5, pady=2)
        
        # Inflammation model path and architecture
        inflam_frame = ttk.LabelFrame(dialog, text="Inflammation Model Configuration")
        inflam_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Path selection
        inflam_path_frame = ttk.Frame(inflam_frame)
        inflam_path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(inflam_path_frame, text="Model Path:").pack(side=tk.LEFT, padx=5)
        inflam_var = tk.StringVar(value=inflammation_path)
        inflam_entry = ttk.Entry(inflam_path_frame, textvariable=inflam_var, width=60)
        inflam_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_inflam():
            path = filedialog.askopenfilename(
                title="Select Inflammation Model",
                filetypes=[("PyTorch Models", "*.pt *.pth"), ("All Files", "*.*")],
                initialdir=str(Path(inflam_var.get()).parent) if inflam_var.get() else "/"
            )
            if path:
                inflam_var.set(path)
        
        ttk.Button(inflam_path_frame, text="Browse...", command=browse_inflam).pack(side=tk.RIGHT, padx=5)
        
        # Architecture selection
        inflam_arch_frame = ttk.Frame(inflam_frame)
        inflam_arch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(inflam_arch_frame, text="Architecture:").pack(side=tk.LEFT, padx=5)
        inflam_arch_var = tk.StringVar(value=inflammation_arch)
        inflam_arch_menu = ttk.Combobox(inflam_arch_frame, textvariable=inflam_arch_var, values=available_architectures)
        inflam_arch_menu.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Add architecture description
        inflam_arch_desc = ttk.Label(inflam_frame, text="Model architecture type (should match your trained model)")
        inflam_arch_desc.pack(fill=tk.X, padx=5, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def use_default():
            # Let AutoClassifier find default models
            self.tissue_model_path = None
            self.inflammation_model_path = None
            self.tissue_model_arch = "resnet18"  # Default architecture
            self.inflammation_model_arch = "resnet18"  # Default architecture
            
            # Update config
            self.config['tissue_model_path'] = ''
            self.config['inflammation_model_path'] = ''
            self.config['tissue_model_architecture'] = 'resnet18'
            self.config['inflammation_model_architecture'] = 'resnet18'
            self.save_config()
            dialog.destroy()
        
        def confirm_paths():
            tissue_path = tissue_var.get()
            inflam_path = inflam_var.get()
            tissue_architecture = tissue_arch_var.get()
            inflam_architecture = inflam_arch_var.get()
            
            if tissue_path and not Path(tissue_path).exists():
                messagebox.showerror("Error", f"Tissue model path does not exist: {tissue_path}")
                return
                
            if inflam_path and not Path(inflam_path).exists():
                messagebox.showerror("Error", f"Inflammation model path does not exist: {inflam_path}")
                return
                
            # Save paths and architectures
            self.tissue_model_path = tissue_path if tissue_path else None
            self.inflammation_model_path = inflam_path if inflam_path else None
            self.tissue_model_arch = tissue_architecture
            self.inflammation_model_arch = inflam_architecture
            
            # Update config
            self.config['tissue_model_path'] = tissue_path
            self.config['inflammation_model_path'] = inflam_path
            self.config['tissue_model_architecture'] = tissue_architecture
            self.config['inflammation_model_architecture'] = inflam_architecture
            self.save_config()
            
            dialog.destroy()
        
        ttk.Button(btn_frame, text="Use Default", command=use_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Confirm", command=confirm_paths).pack(side=tk.RIGHT, padx=5)
        
        # Wait for user to make selection
        self.root.wait_window(dialog)

    def select_annotation_folder(self):
        """Allow user to select a folder for annotations at startup"""
        # Try to read last used folder from config file
        config_file = Path("annotation_config.json")
        last_folder = None
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    last_folder = config.get('annotation_folder')
            except Exception as e:
                logging.error(f"Error reading config file: {e}")
        
        # Create dialog to select folder
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Annotation Folder")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        window_width = 500
        window_height = 200
        position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
        position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
        dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Path display
        path_var = tk.StringVar()
        if last_folder:
            path_var.set(last_folder)
        else:
            path_var.set(str(Path("annotations").absolute()))
        
        ttk.Label(dialog, text="Select folder for storing annotations:").pack(padx=10, pady=10)
        path_entry = ttk.Entry(dialog, textvariable=path_var, width=60)
        path_entry.pack(padx=10, pady=5, fill=tk.X)
        
        def browse_folder():
            folder = filedialog.askdirectory(
                title="Select Annotation Folder",
                initialdir=path_var.get() if path_var.get() else "/"
            )
            if folder:
                path_var.set(folder)
        
        def confirm_folder():
            selected_folder = path_var.get()
            if selected_folder:
                # Create folder if it doesn't exist
                folder_path = Path(selected_folder)
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Create subfolders
                (folder_path / "backups").mkdir(exist_ok=True)
                (folder_path / "clusters").mkdir(exist_ok=True)
                
                # Save to config
                try:
                    with open(config_file, 'w') as f:
                        json.dump({'annotation_folder': selected_folder}, f)
                except Exception as e:
                    logging.error(f"Error saving config: {e}")
                
                # Store in instance
                self.annotation_folder = folder_path
                dialog.destroy()
        
        # Use default annotations folder
        def use_default():
            default_folder = Path("annotations").absolute()
            default_folder.mkdir(exist_ok=True)
            (default_folder / "backups").mkdir(exist_ok=True)
            (default_folder / "clusters").mkdir(exist_ok=True)
            self.annotation_folder = default_folder
            dialog.destroy()
        
        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="Browse...", command=browse_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Use Default", command=use_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Confirm", command=confirm_folder).pack(side=tk.RIGHT, padx=5)
        
        # Wait for user selection
        self.root.wait_window(dialog)
        
        # If no folder was selected, use default
        if not hasattr(self, 'annotation_folder'):
            default_folder = Path("annotations").absolute()
            default_folder.mkdir(exist_ok=True)
            (default_folder / "backups").mkdir(exist_ok=True)
            (default_folder / "clusters").mkdir(exist_ok=True)
            self.annotation_folder = default_folder

    def setup_auto_backup(self):
        """Setup automatic backup of annotations"""
        def backup():
            if hasattr(self, 'current_wsi_name') and self.current_wsi_name and self.annotations:
                try:
                    backup_dir = self.annotation_folder / "backups"
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = backup_dir / f"{self.current_wsi_name}_{timestamp}.json"
                    
                    with open(backup_file, 'w') as f:
                        json.dump(self.annotations, f, indent=2)
                    
                    # Clean old backups (keep last 5)
                    backup_files = sorted(backup_dir.glob(f"{self.current_wsi_name}_*.json"))
                    while len(backup_files) > 5:
                        backup_files[0].unlink()
                        backup_files = backup_files[1:]
                    
                    logging.info(f"Created backup: {backup_file}")
                    
                except Exception as e:
                    logging.error(f"Error creating backup: {str(e)}")
            
            # Schedule next backup
            self.root.after(self.backup_interval, backup)
        
        # Start backup cycle
        self.root.after(self.backup_interval, backup)

    def setup_canvas(self, parent):
        """Setup the canvas with all necessary scrolling and viewport controls"""
        # Create canvas frame
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollbars
        self.h_scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.DEFAULT_WIDTH,
            height=self.DEFAULT_HEIGHT,
            background='#E0E0E0',
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure scrollbars
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        
        # Initialize viewport tracking
        self.canvas_update_after = None
        
        # Bind scroll events based on platform
        if sys.platform == "darwin":  # macOS
            self.canvas.bind("<MouseWheel>", self._handle_trackpad_scroll)
            self.canvas.bind("<Shift-MouseWheel>", self._handle_trackpad_scroll_horizontal)
        else:  # Windows/Linux
            self.canvas.bind("<MouseWheel>", self._handle_trackpad_scroll)
            self.canvas.bind("<Shift-MouseWheel>", self._handle_trackpad_scroll_horizontal)
            self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        
        # Pan with middle mouse button
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, '_pan_start', None))

        
        # Viewport update bindings
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.h_scrollbar.bind("<ButtonRelease-1>", self.update_viewport)
        self.v_scrollbar.bind("<ButtonRelease-1>", self.update_viewport)
        
        # Drawing and annotation bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.continue_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
        # Selection and movement bindings
        self.canvas.bind("<Button-3>", self.select_annotation)  # Right click
        self.canvas.bind("<Control-Button-1>", self.start_move)  # Ctrl + left click
        self.canvas.bind("<Control-B1-Motion>", self.move_annotation)
        self.canvas.bind("<Control-ButtonRelease-1>", self.end_move)
        

    def setup_ui(self):
        """Setup the user interface with all components"""
        # Main horizontal split
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left side - WSI list with downsample factor
        self.wsi_list = WSIListFrame(
            self.paned_window, 
            self.load_selected_wsi,
            downsample=self.DOWNSAMPLE,
            annotation_folder=self.annotation_folder  # Pass the selected folder
        )
        self.paned_window.add(self.wsi_list, weight=1)
        
        # Create right side container
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=15)
        
        # Add current WSI label at the top
        self.current_wsi_label = ttk.Label(
            self.right_frame,
            text="No WSI Selected",
            font=("Arial", 12, "bold")
        )
        self.current_wsi_label.pack(side=tk.TOP, pady=5)
        
        # Create horizontal split for right side
        right_paned = ttk.PanedWindow(self.right_frame, orient=tk.HORIZONTAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Main content (canvas and controls)
        main_content = ttk.Frame(right_paned)
        right_paned.add(main_content, weight=20)
        
        # Annotation list panel
        self.annotation_list_frame = self.setup_annotation_list(right_paned)
        right_paned.add(self.annotation_list_frame, weight=1)
        
        # Setup main content
        self.setup_top_panel(main_content)
        self.setup_canvas(main_content)
        self.setup_status_bar(main_content)
        

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for faster annotation with MacOS support"""
        
        # Additional shortcuts
        self.root.bind("<Return>", self.handle_enter_key)
        self.root.bind("<Delete>", lambda e: self.delete_selected_from_list())
        self.root.bind("e", lambda e: self.edit_selected_annotations(e))
        self.root.bind("<Escape>", lambda e: self.clear_current())

        # Cluster shortcut
        self.root.bind("c", self.handle_cluster_key)
        
        # Display help for keyboard shortcuts
        self.root.bind("<F1>", self.show_keyboard_shortcut_help)
        
        # Quick save for tissue types - using lambda to keep event object
        self.root.bind("1", lambda e: self.handle_tissue_hotkey("corpus", e))
        self.root.bind("2", lambda e: self.handle_tissue_hotkey("antrum", e))
        self.root.bind("3", lambda e: self.handle_tissue_hotkey("intermediate", e))
        self.root.bind("4", lambda e: self.handle_tissue_hotkey("other", e))

    def handle_tissue_hotkey(self, tissue_type, event):
        """Handle tissue type hotkeys (1-4) safely"""
        if self.drawing and self.current_points and len(self.current_points) >= 3:
            # If actively drawing, safely end the drawing first
            first = self.current_points[0]
            last = self.current_points[-1]
            self.canvas.create_line(
                last[0], last[1],
                first[0], first[1],
                fill='red',
                width=2,
                tags='current'
            )
            # Now save with the specified tissue type
            self.quick_save_annotation(tissue_type=tissue_type)
        else:
            # If not actively drawing, just process normally
            self.quick_save_annotation(tissue_type=tissue_type)

    def show_keyboard_shortcut_help(self, event=None):
        """Show help for keyboard shortcuts"""
        help_text = """
        Keyboard Shortcuts:
        
        Drawing & Saving:
        Enter - Quick save current annotation
        Escape - Cancel current drawing
        1 - Quick save as Corpus
        2 - Quick save as Antrum
        3 - Quick save as Intermediate
        4 - Quick save as Other
        
        Editing:
        E - Edit selected annotation(s)
        Delete - Delete selected annotation(s)
        
        Clusters:
        C - Start/complete cluster (click twice)
        
        Navigation:
        Mouse wheel - Scroll vertically
        Shift + Mouse wheel - Scroll horizontally
        Middle mouse - Pan the view
        
        Other:
        F1 - Show this help
        """
        
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("Keyboard Shortcuts")
        help_dialog.transient(self.root)
        help_dialog.grab_set()
        
        # Center dialog
        window_width = 400
        window_height = 500
        position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
        position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
        help_dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Add text with scrollbar
        text_frame = ttk.Frame(help_dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        ttk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)
        
        # Bind escape to close
        help_dialog.bind("<Escape>", lambda e: help_dialog.destroy())

    def handle_enter_key(self, event):
        """Handle Enter key press based on context"""
        # If we have a dialog open, let it handle its own Enter key
        if hasattr(self, 'current_dialog') and self.current_dialog.winfo_exists():
            return
        
        # If no dialog is open, handle quick save
        if self.current_points and len(self.current_points) >= 3:
            self.quick_save_annotation()
            return "break"
        return "break"
    
    def handle_cluster_key(self, event):
        """Handle cluster key press."""
        if not self.current_wsi_name:
            self.status_var.set("No WSI loaded")
            return

        # Get absolute mouse coordinates relative to screen
        root_x, root_y = self.root.winfo_pointerxy()
        
        # Convert to canvas coordinates by finding canvas position relative to root
        canvas_x = root_x - self.canvas.winfo_rootx() + self.canvas.canvasx(0)
        canvas_y = root_y - self.canvas.winfo_rooty() + self.canvas.canvasy(0)


        if self.cluster_start is None:
            # First corner
            self.start_cluster(canvas_x, canvas_y)
        else:
            # Second corner, complete the cluster
            self.complete_cluster(canvas_x, canvas_y)


    def start_cluster(self, canvas_x, canvas_y):
        """Start a new cluster by setting the first corner at mouse position"""
        # Convert to WSI coordinates (unscaled)
        x = canvas_x / self.current_scale
        y = canvas_y / self.current_scale
        
        # Fully clear previous state
        self.canvas.delete('cluster_marker')
        self.cluster_start = (x, y)
        
        self.status_var.set("Cluster started - Press 'c' again to complete")
        
        # Draw marker at exact cursor position
        marker_size = 5  # Increased size for visibility
        self.canvas.create_oval(
            canvas_x - marker_size, canvas_y - marker_size,
            canvas_x + marker_size, canvas_y + marker_size,
            fill='yellow',
            outline='black',
            width=2,
            tags='cluster_marker'
        )

    def complete_cluster(self, canvas_x, canvas_y):
        """Complete the cluster with the second corner at mouse position and automatically process"""
        if self.cluster_start is None:
            self.status_var.set("Error: No cluster start point found")
            return
                
        # Convert to WSI coordinates (unscaled)
        x = canvas_x / self.current_scale
        y = canvas_y / self.current_scale
        
        # Get original start coordinates (unscaled)
        start_x, start_y = self.cluster_start
        
        # Create cluster bounds in full resolution coordinates
        left = min(start_x, x) * self.DOWNSAMPLE
        right = max(start_x, x) * self.DOWNSAMPLE
        top = min(start_y, y) * self.DOWNSAMPLE
        bottom = max(start_y, y) * self.DOWNSAMPLE
        
        # Create cluster object with full resolution coordinates
        cluster = {
            "id": len(self.clusters) + 1,
            "bounds": {
                "left": left,
                "right": right,
                "top": top,
                "bottom": bottom
            }
        }
        
        self.clusters.append(cluster)
        
        # Draw rectangle for cluster at correct display coordinates
        self.canvas.create_rectangle(
            start_x * self.current_scale, 
            start_y * self.current_scale,
            x * self.current_scale, 
            y * self.current_scale,
            outline='yellow',
            width=2,
            tags='cluster_bounds'
        )
        
        # Save clusters to file
        self.save_clusters()
        
        # Automatically process clusters - no button needed anymore
        if hasattr(self, 'current_wsi_name') and self.current_wsi_name:
            self.process_annotations_with_clusters(self.current_wsi_name, self.DOWNSAMPLE)

            # Update the annotation list to show current cluster IDs
            self.update_annotation_list()
        
        # CRITICAL: Reset all state for next cluster
        self.canvas.delete('cluster_marker')
        self.cluster_start = None  # Explicitly set to None
        
        self.status_var.set(f"Cluster {cluster['id']} saved and processed automatically")

    def save_clusters(self):
        """Save clusters to JSON file"""
        if not self.current_wsi_name:
            return
            
        try:
            # Ensure directory exists
            cluster_dir = self.annotation_folder / "clusters"
            cluster_dir.mkdir(exist_ok=True)
            
            cluster_file = cluster_dir / f"{self.current_wsi_name}_clusters.json"
            
            with open(cluster_file, 'w') as f:
                json.dump(self.clusters, f, indent=2)
                
            # Update the WSI list display
            self.wsi_list.update_cluster_display(self.current_wsi_name)
                    
        except Exception as e:
            logging.error(f"Error saving clusters: {e}")

    def process_annotations_with_clusters(self, wsi_name: str, downsample_factor: int = 16):
        """Process annotations and add cluster IDs"""
        try:
            # Load clusters (coordinates are in full resolution)
            cluster_file = self.annotation_folder / "clusters" / f"{wsi_name}_clusters.json"
            if not cluster_file.exists():
                logging.error(f"No clusters found for {wsi_name}")
                return
                
            with open(cluster_file, 'r') as f:
                clusters = json.load(f)
            
            # Process in-memory annotations (for immediate UI update)
            annotations_updated = 0
            
            for annotation in self.annotations:
                # Get center point of annotation in display coordinates
                coords = annotation["geometry"]["coordinates"][0]
                center_x_display = sum(x for x, _ in coords) / len(coords)
                center_y_display = sum(y for _, y in coords) / len(coords)
                
                # Convert to full resolution coordinates
                center_x_full = center_x_display * self.DOWNSAMPLE / self.current_scale
                center_y_full = center_y_display * self.DOWNSAMPLE / self.current_scale
                
                # Find which cluster contains this point
                found_cluster = False
                for cluster in clusters:
                    bounds = cluster["bounds"]
                    if (bounds["left"] <= center_x_full <= bounds["right"] and
                        bounds["top"] <= center_y_full <= bounds["bottom"]):
                        # Add cluster ID to annotation properties
                        annotation["properties"]["cluster_id"] = cluster["id"]
                        found_cluster = True
                        annotations_updated += 1
                        break
                        
                if not found_cluster:
                    # If not in any cluster, mark as unclustered
                    annotation["properties"]["cluster_id"] = None
            
            # Now update the on-disk annotations to ensure persistence
            # First, load the on-disk annotations
            annotation_file = self.annotation_folder / f"{wsi_name}_annotations.json"
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    disk_annotations = json.load(f)
                    
                # Process each on-disk annotation
                for annotation in disk_annotations:
                    # Get center point of annotation (coordinates are in full resolution)
                    coords = annotation["geometry"]["coordinates"][0]
                    center_x = sum(x for x, _ in coords) / len(coords)
                    center_y = sum(y for _, y in coords) / len(coords)
                    
                    # Find which cluster contains this point
                    for cluster in clusters:
                        bounds = cluster["bounds"]
                        if (bounds["left"] <= center_x <= bounds["right"] and
                            bounds["top"] <= center_y <= bounds["bottom"]):
                            # Add cluster ID to annotation properties
                            annotation["properties"]["cluster_id"] = cluster["id"]
                            break
                    else:
                        # If not in any cluster, mark as unclustered
                        annotation["properties"]["cluster_id"] = None
                        
                # Save updated on-disk annotations
                with open(annotation_file, 'w') as f:
                    json.dump(disk_annotations, f, indent=2)
                    
            logging.info(f"Successfully processed {annotations_updated} annotations with clusters")
            self.status_var.set(f"Assigned {annotations_updated} annotations to clusters")
                
        except Exception as e:
            logging.error(f"Error processing clusters: {e}")
            self.status_var.set("Error processing clusters")
            
        return annotations_updated  # Return count for potential use by callers

    def process_current_clusters(self):
        """Process clusters for current WSI"""
        if not self.current_wsi_name:
            messagebox.showwarning("Warning", "Please load a WSI first")
            return
            
        try:
            # Process clusters
            self.process_annotations_with_clusters()
            
            # Reload annotations to show updated cluster IDs
            self.update_annotation_list()
            
            self.status_var.set(f"Processed clusters for {self.current_wsi_name}")
            
        except Exception as e:
            logging.error(f"Error processing clusters: {e}")
            messagebox.showerror("Error", "Failed to process clusters")

    def select_annotation(self, event):
        """Handle right-click selection of annotations"""
        if not self.current_image:
            return
            
        # Get canvas coordinates considering scroll position
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Create click point - coordinates are already in display space (downsampled)
        click_point = Point(canvas_x, canvas_y)
        
        selected = None
        
        # Check each annotation
        for annotation in reversed(self.annotations):  # Reverse to check top-most annotations first
            try:
                # Get coordinates - these are already in display space (downsampled)
                coords = annotation['geometry']['coordinates'][0]
                
                # Ensure coordinates are in the correct format
                coords_tuples = [(float(coord[0]), float(coord[1])) for coord in coords]
                
                # Create polygon with display coordinates
                poly = Polygon(coords_tuples)
                
                # Check if point is inside polygon
                if poly.contains(click_point):
                    selected = annotation
                    break
                    
            except Exception as e:
                logging.error(f"Error during annotation selection: {str(e)}")
                continue
        
        # If an annotation was selected, highlight it and show dialog
        if selected:
            self.selected_annotation = selected
            self.highlight_selected_annotation(selected)
            self.show_annotation_dialog(selected)
            self.status_var.set("Selected annotation")
        else:
            # Clear previous selection if clicking outside
            if hasattr(self, 'selected_annotation'):
                delattr(self, 'selected_annotation')
                self.canvas.delete('highlight')
                self.status_var.set("Selection cleared")

    def process_annotations_with_clusters(self, wsi_name: str, downsample_factor: int = 16):
        """Process annotations and add cluster IDs"""
        try:
            # Load clusters (coordinates are in full resolution)
            cluster_file = self.annotation_folder / "clusters" / f"{wsi_name}_clusters.json"
            if not cluster_file.exists():
                logging.error(f"No clusters found for {wsi_name}")
                return
                
            with open(cluster_file, 'r') as f:
                clusters = json.load(f)
                
            # Load annotations
            annotation_file = self.annotation_folder / f"{wsi_name}_annotations.json"
            if not annotation_file.exists():
                logging.error(f"No annotations found for {wsi_name}")
                return
                
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                
            # Process each annotation
            for annotation in annotations:
                # Get center point of annotation (coordinates are in full resolution)
                coords = annotation["geometry"]["coordinates"][0]
                center_x = sum(x for x, _ in coords) / len(coords)
                center_y = sum(y for _, y in coords) / len(coords)
                
                # Find which cluster contains this point
                for cluster in clusters:
                    bounds = cluster["bounds"]
                    if (bounds["left"] <= center_x <= bounds["right"] and
                        bounds["top"] <= center_y <= bounds["bottom"]):
                        # Add cluster ID to annotation properties
                        annotation["properties"]["cluster_id"] = cluster["id"]
                        break
                else:
                    # If not in any cluster, mark as unclustered
                    annotation["properties"]["cluster_id"] = None
                    
            # Save updated annotations
            with open(annotation_file, 'w') as f:
                json.dump(annotations, f, indent=2)
                
            logging.info(f"Successfully processed clusters for {wsi_name}")
                
        except Exception as e:
            logging.error(f"Error processing clusters: {e}")
    
    def edit_selected_annotations(self, event=None):
        """Edit currently selected annotations using 'e' key"""
        selections = self.annotation_listbox.curselection()
        if selections:
            self.edit_selected_from_list()
        elif hasattr(self, 'selected_annotation'):
            self.show_annotation_dialog(self.selected_annotation)
        else:
            self.status_var.set("No annotation selected for editing")

    def start_move(self, event):
        """Start moving the selected annotation"""
        try:
            # Get canvas coordinates considering scroll position
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            click_point = Point(canvas_x, canvas_y)
            
            # Find the annotation to move
            for annotation in reversed(self.annotations):
                coords = annotation['geometry']['coordinates'][0]
                coords_tuples = [(float(coord[0]), float(coord[1])) for coord in coords]
                poly = Polygon(coords_tuples)
                
                if poly.contains(click_point):
                    self.selected_annotation = annotation
                    # Store initial mouse position
                    self.move_start_x = canvas_x
                    self.move_start_y = canvas_y
                    # Store initial annotation coordinates
                    self.original_coords = annotation['geometry']['coordinates'][0].copy()
                    # Change cursor to indicate movement
                    self.canvas.config(cursor="fleur")
                    break
        except Exception as e:
            logging.error(f"Error starting move: {str(e)}")

    def move_annotation(self, event):
        """Move the selected annotation with the mouse"""
        if hasattr(self, 'selected_annotation') and hasattr(self, 'move_start_x'):
            try:
                # Calculate movement delta
                current_x = self.canvas.canvasx(event.x)
                current_y = self.canvas.canvasy(event.y)
                delta_x = current_x - self.move_start_x
                delta_y = current_y - self.move_start_y
                
                # Update coordinates
                new_coords = []
                for x, y in self.original_coords:
                    new_x = float(x) + delta_x
                    new_y = float(y) + delta_y
                    new_coords.append([new_x, new_y])
                
                # Update annotation coordinates
                self.selected_annotation['geometry']['coordinates'] = [new_coords]
                
                # Redraw all annotations and highlight the selected one
                self.draw_saved_annotations()
                self.highlight_selected_annotation(self.selected_annotation)
                
            except Exception as e:
                logging.error(f"Error moving annotation: {str(e)}")

    def quick_save_annotation(self, event=None, tissue_type=None, inflammation_status=None):
        """Save annotation without dialog using last used or default values"""
        if not self.current_points or len(self.current_points) < 3:
            return
            
        # Use provided values, last used values, or defaults
        tissue_type = tissue_type or getattr(self, 'last_tissue_type', list(self.ANNOTATION_OPTIONS['tissue'].keys())[0])
        inflammation_status = inflammation_status or getattr(self, 'default_inflammation_status', 
                                                    list(self.ANNOTATION_OPTIONS['inflammation_status'].keys())[0])
        
        # Save the annotation
        self.save_current_annotation(tissue_type, inflammation_status)
        
        # Store as last used values
        self.last_tissue_type = tissue_type
        
        # Status feedback showing what was saved
        self.status_var.set(f"Quick saved: {tissue_type} - {inflammation_status}")
        
        # Ensure drawing state is reset
        self.drawing = False
        
        # Make sure the background image is redrawn if it was lost
        if self.current_image and not self.canvas.find_withtag('background'):
            self.canvas.create_image(
                0, 0,
                image=self.current_image,
                anchor="nw",
                tags='background'
            )
            # Make sure background is at the bottom of the stack
            self.canvas.tag_lower('background')

    def _bind_scrolling(self):
        """Bind all scrolling events"""
        # Mouse wheel scrolling
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self._on_linux_scroll)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_linux_scroll)  # Linux scroll down
        
        # Shift + scroll for horizontal scrolling
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        
        # Track pad scrolling for macOS
        self.canvas.bind("<Motion>", self._bind_trackpad)
        self.canvas.bind("<Leave>", self._unbind_trackpad)
        
        # Pan with middle mouse button
        self.canvas.bind("<Button-2>", self._start_pan)
        self.canvas.bind("<B2-Motion>", self._pan)
        self.canvas.bind("<ButtonRelease-2>", lambda e: setattr(self, '_pan_start', None))

    def _bind_trackpad(self, event=None):
        """Bind trackpad scrolling when mouse enters canvas"""
        # Bind to root to capture events outside canvas
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _unbind_trackpad(self, event=None):
        """Unbind trackpad scrolling when mouse leaves canvas"""
        self.root.unbind("<MouseWheel>")
        self.root.unbind("<Shift-MouseWheel>")

    def _on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling"""
        delta = event.delta
        
        # Handle macOS large delta values
        if abs(delta) > 100:
            delta = int(delta/120)
        
        self.canvas.yview_scroll(int(-1 * delta), "units")
        self.schedule_viewport_update()
        
        # Prevent event propagation
        return "break"

    def _on_shift_mousewheel(self, event):
        """Handle horizontal scrolling with Shift + mouse wheel"""
        delta = event.delta
        
        # Handle macOS large delta values
        if abs(delta) > 100:
            delta = int(delta/120)
        
        self.canvas.xview_scroll(int(-1 * delta), "units")
        self.schedule_viewport_update()
        
        # Prevent event propagation
        return "break"

    def _on_linux_scroll(self, event):
        """Handle Linux-style scroll events"""
        if event.num == 4:  # Scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Scroll down
            self.canvas.yview_scroll(1, "units")
        
        self.schedule_viewport_update()
        return "break"

    def _start_pan(self, event):
        """Start panning with middle mouse button"""
        self.canvas.config(cursor="fleur")
        self._pan_start = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)

    def _pan(self, event):
        """Pan the canvas with middle mouse button"""
        if hasattr(self, '_pan_start'):
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.schedule_viewport_update()

    def schedule_viewport_update(self, delay=100):
        """Schedule a viewport update with debouncing"""
        if self.canvas_update_after:
            self.root.after_cancel(self.canvas_update_after)
        self.canvas_update_after = self.root.after(delay, self.update_viewport)

    def update_viewport(self, event=None):
        """Update the viewport overlay on the thumbnail"""
        if not hasattr(self, 'current_wsi_name') or not self.current_wsi_name:
            return
                
        try:
            # Get current view position (these are fractions between 0 and 1)
            x_view = self.canvas.xview()
            y_view = self.canvas.yview()
            
            # Get viewport dimensions
            viewport_width = self.canvas.winfo_width()
            viewport_height = self.canvas.winfo_height()
            
            # Get total scrollable area
            scroll_region = self.canvas.bbox("all")
            if not scroll_region:
                return
            
            total_width = scroll_region[2] - scroll_region[0]
            total_height = scroll_region[3] - scroll_region[1]
            
            # Convert scroll fractions to actual coordinates
            x_scroll = x_view[0] * total_width
            y_scroll = y_view[0] * total_height
            
            # Update viewport in WSI list
            self.wsi_list.update_viewport(
                self.current_wsi_name,
                (x_view[0], y_view[0], viewport_width/total_width, viewport_height/total_height)
            )
                
        except Exception as e:
            logging.error(f"Error updating viewport: {str(e)}")

    def setup_annotation_list(self, parent):
        """Setup the annotation list panel with multi-select support"""
        frame = ttk.LabelFrame(parent, text="Annotations")
        
        # Create listbox with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Change selectmode to EXTENDED for multi-select
        self.annotation_listbox = tk.Listbox(
            list_frame, 
            selectmode=tk.EXTENDED,  # Allow multiple selections
            width=25
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        self.annotation_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons for managing annotations
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            btn_frame, 
            text="Delete", 
            command=self.delete_selected_from_list
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            btn_frame, 
            text="Edit",
            command=self.edit_selected_from_list
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            btn_frame,
            text="Show Help (F1)",
            command=self.show_keyboard_shortcut_help
        ).pack(side=tk.LEFT, padx=2)
        
        # Add selection counter label
        self.selection_label = ttk.Label(frame, text="0 annotations selected")
        self.selection_label.pack(pady=2)
        
        # Bind selection event for updating counter and highlighting
        self.annotation_listbox.bind('<<ListboxSelect>>', self.on_annotation_select)
        
        return frame
        
    def update_annotation_list(self):
        """Update the annotation listbox"""
        # Clear existing items
        self.annotation_listbox.delete(0, tk.END)
        
        # Add current annotations
        for idx, annotation in enumerate(self.annotations, 1):
            tissue_type = annotation['properties']['classification']['tissue_type']
            
            # Handle both old and new naming convention for backward compatibility
            inflammation_status = annotation['properties']['classification'].get('inflammation_status', 
                                 annotation['properties']['classification'].get('inflammation_status', '-'))
            
            cluster_id = annotation['properties'].get('cluster_id', '-')
            
            self.annotation_listbox.insert(tk.END, 
                f"{idx}. {tissue_type} - {inflammation_status}  | ({cluster_id})")

    def center_on_annotation(self, annotation):
        """Center the view on the selected annotation"""
        try:
            # Get coordinates of the annotation
            coords = annotation['geometry']['coordinates'][0]
            
            # Calculate center point of the annotation
            x_coords = [x for x, y in coords]
            y_coords = [y for x, y in coords]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Get scroll region
            scroll_region = self.canvas.bbox("all")
            if not scroll_region:
                return
                
            total_width = scroll_region[2] - scroll_region[0]
            total_height = scroll_region[3] - scroll_region[1]
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Calculate scroll fractions
            x_fraction = (center_x - canvas_width/2) / total_width
            y_fraction = (center_y - canvas_height/2) / total_height
            
            # Clamp values between 0 and 1
            x_fraction = max(0.0, min(1.0, x_fraction))
            y_fraction = max(0.0, min(1.0, y_fraction))
            
            # Move view to center on annotation
            self.canvas.xview_moveto(x_fraction)
            self.canvas.yview_moveto(y_fraction)
            
            # Update viewport
            self.schedule_viewport_update(delay=50)
            
        except Exception as e:
            logging.error(f"Error centering on annotation: {str(e)}")

    def on_annotation_select(self, event):
        """Handle selection in the annotation list"""
        selections = self.annotation_listbox.curselection()
        num_selected = len(selections)
        
        # Update selection counter
        self.selection_label.config(text=f"{num_selected} annotation{'s' if num_selected != 1 else ''} selected")
        
        # Clear previous highlights
        self.canvas.delete('highlight')
        
        if num_selected > 0:
            # Highlight all selected annotations
            for idx in selections:
                annotation = self.annotations[idx]
                self.highlight_selected_annotation(annotation, clear_previous=False)
                
            # Center view on first selected annotation
            first_selected = self.annotations[selections[0]]
            self.center_on_annotation(first_selected)


    def delete_selected_from_list(self):
        """Delete multiple selected annotations from the list"""
        selections = self.annotation_listbox.curselection()
        if not selections:
            messagebox.showwarning("Warning", "Please select annotations to delete")
            return
        
        num_selected = len(selections)
        result = messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete {num_selected} annotation{'s' if num_selected != 1 else ''}?"
        )
        
        if result:
            try:
                # Convert to list and sort in reverse order to preserve indices
                indices = sorted(list(selections), reverse=True)
                
                # Remove annotations
                for idx in indices:
                    self.annotations.pop(idx)
                
                # Clear highlights
                self.canvas.delete('highlight')
                
                # Update display
                self.draw_saved_annotations()
                self.update_annotation_list()
                
                # Save annotations first
                self.save_annotations()
                
                # Force a refresh of the count by explicitly updating after save
                if hasattr(self, 'current_wsi_name'):
                    self.wsi_list.update_annotation_count(self.current_wsi_name)
                
                # Update status
                self.status_var.set(f"Deleted {num_selected} annotation{'s' if num_selected != 1 else ''}")
                
                # Clear selection label
                self.selection_label.config(text="0 annotations selected")
                
            except Exception as e:
                logging.error(f"Error during multiple deletion: {str(e)}")
                messagebox.showerror("Error", "Failed to delete some annotations")

    def edit_selected_from_list(self):
        """Edit selected annotations"""
        selections = self.annotation_listbox.curselection()
        if not selections:
            messagebox.showwarning("Warning", "Please select annotations to edit")
            return
        
        # Show dialog for bulk editing
        self.show_bulk_edit_dialog([self.annotations[idx] for idx in selections])

    def show_bulk_edit_dialog(self, annotations_to_edit):
        """Show dialog for editing annotations at once"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit annotation{'s' if len(annotations_to_edit) != 1 else ''}")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Store dialog state
        self.current_dialog = dialog
        self.dialog_saved = False
        
        # Tissue selection
        tissue_frame = ttk.LabelFrame(dialog, text="Tissue Type")
        tissue_frame.pack(padx=10, pady=5, fill=tk.X)
        
        tissue_options = list(self.ANNOTATION_OPTIONS['tissue'].keys())
        selected_tissue = tk.StringVar(value=tissue_options[0])
        ttk.Label(tissue_frame, text="Tissue:").pack(side=tk.LEFT, padx=5)
        ttk.OptionMenu(tissue_frame, selected_tissue, 
                    tissue_options[0], *tissue_options).pack(side=tk.LEFT, padx=5)
        
        # Inflammation selection
        inflammation_frame = ttk.LabelFrame(dialog, text="Inflammation Status")
        inflammation_frame.pack(padx=10, pady=5, fill=tk.X)
        
        inflammation_options = list(self.ANNOTATION_OPTIONS['inflammation_status'].keys())
        selected_inflammation = tk.StringVar(value=inflammation_options[0])
        ttk.Label(inflammation_frame, text="Inflammation:").pack(side=tk.LEFT, padx=5)
        ttk.OptionMenu(inflammation_frame, selected_inflammation,
                    inflammation_options[0], *inflammation_options).pack(side=tk.LEFT, padx=5)
        
        # Add count label
        count_label = ttk.Label(
            dialog, 
            text=f"Editing {len(annotations_to_edit)} annotation{'s' if len(annotations_to_edit) != 1 else ''}"
        )
        count_label.pack(pady=5)
        
        def save_and_close(event=None):
            """Apply changes to all selected annotations"""
            tissue_type = selected_tissue.get()
            inflammation_status = selected_inflammation.get()
            
            # Update all selected annotations
            for annotation in annotations_to_edit:
                self.update_annotation(annotation, tissue_type, inflammation_status)
            
            self.dialog_saved = True
            dialog.destroy()
        
        def on_dialog_close():
            """Handle dialog closing"""
            dialog.destroy()
        
        # Create a frame to hold the buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        # Add Cancel button on the left
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_dialog_close)
        cancel_button.pack(side='left', padx=2)

        # Add Save button on the right
        save_button = ttk.Button(button_frame, text="Save", command=save_and_close)
        save_button.pack(side='left', padx=2)

        # Bind closing event
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)

        # Bind Enter key to save
        dialog.bind("<Return>", save_and_close)
        
        # Bind Escape key to cancel
        dialog.bind("<Escape>", lambda e: on_dialog_close())

    def set_slide_wide_inflammation_status(self, status=None):
        """Set inflammation status for all annotations in the current slide"""
        if not self.current_wsi_name:
            messagebox.showwarning("Warning", "No slide loaded")
            return
            
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to update")
            return
            
        # If status not provided, show a selection dialog
        if status is None:
            status_dialog = tk.Toplevel(self.root)
            status_dialog.title("Set Slide-Wide Inflammation Status")
            status_dialog.transient(self.root)
            status_dialog.grab_set()
            
            # Center dialog
            window_width = 300
            window_height = 150
            position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
            position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
            status_dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
            
            ttk.Label(status_dialog, text="Select inflammation status for all annotations:").pack(pady=10)
            
            selected_status = tk.StringVar()
            selected_status.set(self.default_inflammation_status)  # Use current default
            
            frame = ttk.Frame(status_dialog)
            frame.pack(fill=tk.X, pady=10)
            
            # Create radio buttons for each status
            for option in self.ANNOTATION_OPTIONS['inflammation_status'].keys():
                ttk.Radiobutton(
                    frame, 
                    text=option.capitalize(),
                    variable=selected_status,
                    value=option
                ).pack(anchor=tk.W, padx=20)
            
            def apply_status():
                # Apply selected status and update default
                status = selected_status.get()
                self._apply_inflammation_status(status)
                status_dialog.destroy()
                
            # Add buttons
            button_frame = ttk.Frame(status_dialog)
            button_frame.pack(fill=tk.X, pady=10)
            
            ttk.Button(button_frame, text="Cancel", command=status_dialog.destroy).pack(side=tk.LEFT, padx=5, expand=True)
            ttk.Button(button_frame, text="Apply", command=apply_status).pack(side=tk.LEFT, padx=5, expand=True)
            
            # Bind Enter key to apply
            status_dialog.bind("<Return>", lambda e: apply_status())
            
            # Bind Escape key to cancel
            status_dialog.bind("<Escape>", lambda e: status_dialog.destroy())
            
        else:
            # If status is provided, apply directly
            self._apply_inflammation_status(status)
            
    def _apply_inflammation_status(self, status):
        """Apply the inflammation status to all annotations"""
        count = 0
        for annotation in self.annotations:
            annotation['properties']['classification']['inflammation_status'] = status
            count += 1
            
        # Update display
        self.draw_saved_annotations()
        self.update_annotation_list()
        
        # Save changes
        self.save_annotations()
        
        # Update status
        self.status_var.set(f"Updated {count} annotations with inflammation status: {status}")
            
    def detect_regions_of_interest(self):
        """Automatically detect regions of interest without classification"""
        if not self.current_image or not hasattr(self, '_current_image_path'):
            messagebox.showwarning("Warning", "Please load a WSI first")
            return
            
        try:
            # Load image directly from file
            img = cv2.imread(str(self._current_image_path))
            if img is None:
                raise ValueError("Failed to load image")
                
            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Create mask for non-white areas (focusing on purple/pink hues common in H&E stains)
            masks = []
            
            # Purple/pink mask
            lower_purple = np.array([130, 20, 50])
            upper_purple = np.array([170, 255, 255])
            purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
            masks.append(purple_mask)
            
            # Blue mask (for darker stains)
            lower_blue = np.array([100, 20, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            masks.append(blue_mask)
            
            # Create black hole mask to exclude them
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            black_mask = (gray < 30).astype(np.uint8) * 255  # Detect very dark regions
            
            # General darkness detection (excluding black holes and grey areas)
            # Tighten the threshold range to exclude light grey areas
            tissue_mask = ((gray < 180) & (gray > 30)).astype(np.uint8) * 255
            masks.append(tissue_mask)
            
            # Combine all masks
            combined_mask = np.zeros_like(purple_mask)
            for mask in masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Remove black holes from the combined mask
            combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(black_mask))
            
            # Clean up mask
            kernel = np.ones((5,5), np.uint8)
            cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
            
            # Add buffer zone around detected regions
            buffer_size = 20  # Adjust this value to change the buffer size
            buffer_kernel = np.ones((buffer_size, buffer_size), np.uint8)
            dilated_mask = cv2.dilate(cleaned_mask, buffer_kernel, iterations=1)
            
            # Additional check to remove any remaining black holes after dilation
            dilated_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(black_mask))
            
            # Find contours on the dilated mask
            contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and process contours
            min_area = 1000 * self.DOWNSAMPLE  # Adjust minimum area for full resolution
            max_aspect_ratio = 10.0  # Maximum allowed aspect ratio (length/width)
            new_annotations = []
            
            for contour in contours:
                # Check area
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                # Get bounding rectangle to check aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(max(w, h)) / min(w, h)
                
                # Skip if aspect ratio is too high (too long and thin)
                if aspect_ratio > max_aspect_ratio:
                    continue
                
                # Get rotated rectangle to check actual orientation
                rect = cv2.minAreaRect(contour)
                box_w, box_h = rect[1]
                if box_w == 0 or box_h == 0:  # Prevent division by zero
                    continue
                    
                rotated_aspect = float(max(box_w, box_h)) / min(box_w, box_h)
                if rotated_aspect > max_aspect_ratio:
                    continue
                
                # Additional check to filter out grey areas
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                roi = cv2.bitwise_and(gray, gray, mask=mask)
                mean_intensity = cv2.mean(roi, mask=mask)[0]
                intensity_std = np.std(roi[mask > 0])
                
                # Skip if region is too grey (high intensity, low variance)
                if mean_intensity > 150 and intensity_std < 20:
                    continue
                
                # Calculate solidity (area / convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area) / hull_area
                    # Skip if region is too sparse
                    if solidity < 0.3:
                        continue
                    
                # Simplify contour while preserving shape
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Smooth the contour to remove jagged edges
                smooth_contour = []
                if len(approx) > 2:
                    # Use a rolling average to smooth the contour
                    for i in range(len(approx)):
                        prev_idx = (i - 1) % len(approx)
                        next_idx = (i + 1) % len(approx)
                        
                        curr_point = approx[i][0]
                        prev_point = approx[prev_idx][0]
                        next_point = approx[next_idx][0]
                        
                        # Calculate average point
                        smooth_x = (prev_point[0] + curr_point[0] + next_point[0]) / 3
                        smooth_y = (prev_point[1] + curr_point[1] + next_point[1]) / 3
                        
                        smooth_contour.append([smooth_x, smooth_y])
                
                # Convert smoothed coordinates to annotation format
                if len(smooth_contour) >= 3:  # Only create if we have at least 3 points
                    # Scale coordinates correctly for display
                    display_coords = [
                        [int(x * self.current_scale),
                        int(y * self.current_scale)]
                        for x, y in smooth_contour
                    ]
                    
                    annotation = {
                        "type": "Feature",
                        "id": str(uuid.uuid4()),
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [display_coords]
                        },
                        "properties": {
                            "objectType": "annotation",
                            "classification": {
                                "tissue_type": "other",  # Default type before classification
                                "inflammation_status": self.default_inflammation_status
                            }
                        }
                    }
                    new_annotations.append(annotation)
            
            # Add new annotations to existing ones
            self.annotations.extend(new_annotations)
            
            # Update display
            self.draw_saved_annotations()
            self.update_annotation_list()
            
            # Autosave
            self.save_annotations()
            
            num_regions = len(new_annotations)
            messagebox.showinfo("Success", 
                f"Added {num_regions} auto-detected regions. Use 'Auto-classify Regions' to classify them.")
            
        except Exception as e:
            logging.error(f"Error in region detection: {str(e)}")
            messagebox.showerror("Error", "Failed to auto-detect regions")

    def classify_regions(self):
        """Classify the detected regions using the loaded models"""
        if not self.current_wsi_name or not self.annotations:
            messagebox.showwarning("Warning", "No WSI or annotations loaded to classify")
            return
        
        # First, select model paths if they haven't been selected yet
        if not hasattr(self, 'tissue_model_path') or self.tissue_model_path is None:
            self.select_model_paths()
        
        # Load classifier if not already loaded
        if not hasattr(self, 'auto_classifier'):
            try:
                # Show loading dialog
                loading_dialog = tk.Toplevel(self.root)
                loading_dialog.title("Loading Models")
                loading_dialog.transient(self.root)
                loading_dialog.grab_set()
                
                # Center dialog
                window_width = 300
                window_height = 100
                position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
                position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
                loading_dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
                
                ttk.Label(loading_dialog, text="Loading classification models...").pack(pady=20)
                self.root.update()
                
                # Import and initialize classifier
                from auto_classifier import AutoClassifier
                self.auto_classifier = AutoClassifier(
                    annotation_options=self.ANNOTATION_OPTIONS,
                    custom_tissue_model=self.tissue_model_path,
                    custom_inflammation_model=self.inflammation_model_path
                )
                
                # Close loading dialog
                loading_dialog.destroy()
                
            except Exception as e:
                loading_dialog.destroy() if 'loading_dialog' in locals() else None
                logging.error(f"Error initializing classifier: {str(e)}")
                messagebox.showerror("Error", "Could not load classification models")
                return
        
        try:
            # Try to find the corresponding MRXS file
            mrxs_paths = []
            
            # Search in the raw slides directory for matching WSI
            if hasattr(self, 'paths') and 'RAW_DIR' in self.paths:
                raw_dir = self.paths['RAW_DIR'] / 'slides'
                if raw_dir.exists():
                    pattern = f"{self.current_wsi_name}*.mrxs"
                    mrxs_paths = list(raw_dir.glob(pattern))
            
            # If not found, try a more generic search
            if not mrxs_paths:
                # Try to get the parent directory of the current image path
                if hasattr(self, '_current_image_path'):
                    image_dir = self._current_image_path.parent.parent
                    if image_dir.exists():
                        pattern = f"**/{self.current_wsi_name}*.mrxs"
                        mrxs_paths = list(image_dir.glob(pattern))
            
            # If still not found, use a fallback approach based on the PNG filename
            if not mrxs_paths and hasattr(self, '_current_image_path'):
                # Extract WSI info from PNG filename
                png_name = self._current_image_path.stem
                base_name = png_name.replace('_downsampled16x', '')
                
                # Try to find it in common directories
                common_dirs = [
                    Path("./data/raw/slides"),
                    Path("/data/slides"),
                    Path("/mnt/data/slides")
                ]
                
                for dir_path in common_dirs:
                    if dir_path.exists():
                        pattern = f"{base_name}*.mrxs"
                        found_paths = list(dir_path.glob(pattern))
                        if found_paths:
                            mrxs_paths = found_paths
                            break
            
            if mrxs_paths:
                mrxs_path = mrxs_paths[0]  # Use the first match
                logging.info(f"Found matching MRXS file: {mrxs_path}")
                
                # Show progress dialog
                progress_dialog = tk.Toplevel(self.root)
                progress_dialog.title("Classifying Regions")
                progress_dialog.transient(self.root)
                progress_dialog.grab_set()
                
                # Center dialog
                window_width = 300
                window_height = 100
                position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
                position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
                progress_dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
                
                progress_label = ttk.Label(progress_dialog, text="Classifying regions...")
                progress_label.pack(pady=10)
                
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(
                    progress_dialog, 
                    orient=tk.HORIZONTAL, 
                    length=250, 
                    mode='determinate',
                    variable=progress_var
                )
                progress_bar.pack(pady=10, padx=20)
                self.root.update()
                
                # Process in batches to allow UI updates
                total_annotations = len(self.annotations)
                batch_size = 10
                for i in range(0, total_annotations, batch_size):
                    batch = self.annotations[i:i+batch_size]
                    
                    # Update progress
                    progress_var.set((i / total_annotations) * 100)
                    progress_label.config(text=f"Classifying regions... {i}/{total_annotations}")
                    progress_dialog.update()
                    
                    # Classify batch
                    classified_batch = self.auto_classifier.process_wsi(mrxs_path, batch)
                    
                    # Update classifications
                    for j, classified in enumerate(classified_batch):
                        idx = i + j
                        if idx < total_annotations:
                            tissue_type = classified['properties']['classification']['tissue_type']
                            self.annotations[idx]['properties']['classification'].update({
                                'tissue_type': tissue_type,
                                'color': self.ANNOTATION_OPTIONS['tissue'][tissue_type]
                            })
                
                # Close progress dialog
                progress_dialog.destroy()
                
                # Update display
                self.draw_saved_annotations()
                self.update_annotation_list()
                
                # Autosave
                self.save_annotations()
                
                self.status_var.set("Regions classified successfully")
                messagebox.showinfo("Success", "All regions have been classified")
                
            else:
                logging.warning(f"Could not find matching MRXS file for {self.current_wsi_name}")
                messagebox.showwarning("Warning", "Could not find matching MRXS file for classification")
                
        except Exception as e:
            logging.error(f"Error during classification: {str(e)}")
            messagebox.showerror("Error", "Failed to classify regions")

    def auto_detect_annotations(self):
        """Alias for detect_regions_of_interest for backward compatibility"""
        self.detect_regions_of_interest()
    
    def setup_top_panel(self, parent):
        """Setup top panel with buttons and inflammation status controls"""
        top_panel = ttk.Frame(parent)
        top_panel.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Left side - file operations
        file_frame = ttk.LabelFrame(top_panel, text="File")
        file_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        ttk.Button(file_frame, text="Load Slide Directory", 
                command=self.load_wsi_directory).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
                
        # Middle - annotation operations
        annotation_frame = ttk.LabelFrame(top_panel, text="Annotations")
        annotation_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Split auto-detect into two separate functions
        ttk.Button(annotation_frame, text="Detect Regions", 
                command=self.detect_regions_of_interest).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
        
        ttk.Button(annotation_frame, text="Auto-classify Regions", 
                command=self.classify_regions).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
        
        
        # Quick inflation status buttons
        inflammation_frame = ttk.LabelFrame(top_panel, text="Quick Inflammation Status")
        inflammation_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Add buttons for quick inflammation status setting
        for status in self.ANNOTATION_OPTIONS['inflammation_status'].keys():
            ttk.Button(
                inflammation_frame,
                text=f"Set All {status.capitalize()}",
                command=lambda s=status: self.set_slide_wide_inflammation_status(s)
            ).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
        
        # Default inflammation label
        self.default_inflammation_label = ttk.Label(
            inflammation_frame, 
            text=f"Default: {self.default_inflammation_status.capitalize()}"
        )
        self.default_inflammation_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Right side - cluster operations  
        cluster_frame = ttk.LabelFrame(top_panel, text="Clusters")
        cluster_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # The "Process Clusters" button is removed as it's now automatic
        
        # Delete all clusters button
        ttk.Button(cluster_frame, text="Delete All Clusters", 
                command=self.delete_all_clusters).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
                
        # Toggle clusters visibility button
        self.toggle_clusters_button = ttk.Button(
            cluster_frame, 
            text="Hide Clusters",
            command=self.toggle_clusters_visibility
        )
        self.toggle_clusters_button.pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)
        
        # Add cluster hotkey reminder
        ttk.Label(
            cluster_frame,
            text="Press 'C' twice to create cluster",
            font=("Arial", 8),
            foreground="gray"
        ).pack(side=tk.TOP, padx=5, pady=2)
        
        # Add auto-process indicator
        ttk.Label(
            cluster_frame,
            text="Clusters process automatically",
            font=("Arial", 8),
            foreground="green"
        ).pack(side=tk.TOP, padx=5, pady=2)
        
        # Help frame
        help_frame = ttk.LabelFrame(top_panel, text="Help")
        help_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        ttk.Button(
            help_frame,
            text="Keyboard Shortcuts (F1)",
            command=self.show_keyboard_shortcut_help
        ).pack(side=tk.TOP, padx=5, pady=2, fill=tk.X)

    def on_canvas_configure(self, event):
        """Handle canvas resize events"""
        self.schedule_viewport_update()
    
    def _on_mousewheel(self, event):
        """Handle vertical scrolling"""
        self.canvas.yview_scroll(int(-0.5 * (event.delta / 120)), "units")

    def _on_shift_mousewheel(self, event):
        """Handle horizontal scrolling"""
        self.canvas.xview_scroll(int(-0.5 * (event.delta / 120)), "units")

    def _start_pan(self, event):
        """Start panning"""
        self.canvas.scan_mark(event.x, event.y)
        self._pan_start = (event.x, event.y)

    def _pan(self, event):
        """Pan the image"""
        if self._pan_start:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    def setup_status_bar(self, parent):
        """Setup the status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add a divider
        ttk.Separator(status_frame, orient='horizontal').pack(fill=tk.X, pady=2)
        
        # Status text
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add keyboard shortcut reminder
        ttk.Label(
            status_frame,
            text="Press F1 for help",
            padding=(5, 2)
        ).pack(side=tk.RIGHT)
    
    def _handle_trackpad_scroll(self, event):
        """Handle trackpad scrolling"""
        delta = event.delta
        
        # Handle macOS large delta values
        if abs(delta) > 100:
            delta = int(delta/120)
        
        self.canvas.yview_scroll(int(-1 * delta), "units")
        self.schedule_viewport_update()
        return "break"

    def _handle_trackpad_scroll_horizontal(self, event):
        """Handle horizontal trackpad scrolling"""
        delta = event.delta
        
        # Handle macOS large delta values
        if abs(delta) > 100:
            delta = int(delta/120)
        
        self.canvas.xview_scroll(int(-1 * delta), "units")
        self.schedule_viewport_update()
        return "break"

    def setup_bindings(self):
        """Setup mouse and keyboard bindings"""
        # Drawing bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.continue_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)
        
        # Selection and movement bindings
        self.canvas.bind("<Button-3>", self.select_annotation)  # Right click to select
        self.canvas.bind("<Button-2>", self.start_move)        # Middle mouse to start move
        self.canvas.bind("<B2-Motion>", self.move_annotation)  # Middle mouse drag to move
        self.canvas.bind("<ButtonRelease-2>", self.end_move)   # Middle mouse release to end move
            
    def load_wsi_directory(self):
        """Load directory containing WSI PNG files with progress indication"""
        directory = filedialog.askdirectory(
            title="Select Directory with WSI PNGs"
        )
        
        if not directory:
            return
        
        try:
            # Clear existing list
            self.wsi_list.clear()
            
            # Find all PNG files
            png_files = sorted(Path(directory).glob("*_downsampled16x.png"))
            total_files = len(png_files)
            
            if total_files == 0:
                messagebox.showinfo("Info", "No WSI files found in selected directory")
                return
            
            # Create progress dialog
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("Loading WSIs")
            self.progress_window.transient(self.root)
            self.progress_window.grab_set()
            
            # Center progress window
            self.progress_window.geometry("300x150")
            progress_x = self.root.winfo_x() + (self.root.winfo_width() - 300) // 2
            progress_y = self.root.winfo_y() + (self.root.winfo_height() - 150) // 2
            self.progress_window.geometry(f"+{progress_x}+{progress_y}")
            
            # Add progress bar and label
            ttk.Label(self.progress_window, text="Loading WSI files...").pack(pady=10)
            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                self.progress_window,
                length=200,
                mode='determinate',
                variable=self.progress_var
            )
            self.progress_bar.pack(pady=10)
            
            self.progress_label = ttk.Label(self.progress_window, text="0 / 0 files")
            self.progress_label.pack(pady=10)
            
            def update_progress(current, total):
                if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                    try:
                        self.progress_var.set((current / total) * 100)
                        self.progress_label.config(text=f"{current} / {total} files")
                        self.progress_window.update()
                    except Exception as e:
                        logging.error(f"Error updating progress: {str(e)}")
            
            def process_files():
                try:
                    for i, png_path in enumerate(png_files, 1):
                        if not hasattr(self, 'progress_window') or not self.progress_window.winfo_exists():
                            break
                        
                        # Extract WSI name
                        wsi_name = png_path.stem.replace('_downsampled16x', '')
                        
                        # Add to list (this now uses lazy loading)
                        self.wsi_list.add_wsi(png_path, wsi_name)
                        
                        # Update progress
                        self.root.after_idle(
                            lambda c=i, t=total_files: update_progress(c, t)
                        )
                    
                    # Close progress window
                    if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                        self.progress_window.destroy()
                        delattr(self, 'progress_window')
                    
                    self.status_var.set(f"Loaded {total_files} WSIs from directory")
                    
                except Exception as e:
                    logging.error(f"Error loading WSI directory: {str(e)}")
                    if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                        self.progress_window.destroy()
                        delattr(self, 'progress_window')
                    messagebox.showerror("Error", f"Error loading WSIs: {str(e)}")
            
            # Start processing in background
            threading.Thread(target=process_files, daemon=True).start()
            
        except Exception as e:
            logging.error(f"Error initiating WSI directory load: {str(e)}")
            messagebox.showerror("Error", f"Error loading WSI directory: {str(e)}")

    def load_selected_wsi(self, png_path: Path):
        """Load selected WSI PNG file with improved performance"""
        try:
            # Reset scrollbars to top-left
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
            
            # Clear existing content and selection
            self.canvas.delete("all")
            self.clear_current()
            self.annotations = []
            if hasattr(self, 'selected_annotation'):
                delattr(self, 'selected_annotation')
            self.annotation_listbox.delete(0, tk.END)
            
            # Calculate center position for loading text
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            center_x = canvas_width // 2
            center_y = canvas_height // 2
            
            # Show loading indicator with background
            loading_bg = self.canvas.create_rectangle(
                0, 0, canvas_width, canvas_height,
                fill="#E0E0E0",
                tags="loading"
            )
            loading_text = self.canvas.create_text(
                center_x, center_y,
                text="Loading...",
                font=("Arial", 24),
                fill="gray",
                tags="loading"
            )
            
            # Update status and force refresh
            self.status_var.set(f"Loading {png_path.stem}...")
            self.root.update()

            # Extract WSI name and store path
            self.current_wsi_name = png_path.stem.replace('_downsampled16x', '')
            self._current_image_path = png_path


            def load_image():
                try:
                    # Clear existing clusters
                    self.clusters = []
                    self.canvas.delete('cluster_bounds')
                    # Check cache first
                    if png_path in self.image_cache:
                        img = self.image_cache[png_path]
                    else:
                        # Load new image with PIL's lazy loading
                        img = Image.open(png_path)
                        
                        # Calculate scale factor to fit image within window, then multiply by 5
                        window_width = self.canvas.winfo_width()
                        window_height = self.canvas.winfo_height()
                        scale_x = window_width / img.width
                        scale_y = window_height / img.height
                        base_scale = min(scale_x, scale_y)
                        scale = min(base_scale * 5, 1.0)  # Scale up 5x but don't exceed original size
                        
                        if scale < 1.0:
                            new_width = int(img.width * scale)
                            new_height = int(img.height * scale)
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Store the actual scale used for annotation coordinate conversion
                        self.current_scale = scale
                        
                        # Manage cache size
                        if len(self.image_cache) >= self.max_cache_size:
                            oldest_key = next(iter(self.image_cache))
                            del self.image_cache[oldest_key]
                        
                        # Add to cache
                        self.image_cache[png_path] = img

                    # Convert to PhotoImage in main thread
                    self.root.after_idle(lambda: self.finalize_image_loading(img, png_path))
                    
                except Exception as e:
                    logging.error(f"Error in background loading: {str(e)}")
                    self.root.after_idle(lambda: self.handle_loading_error(str(e)))

            # Start loading in background
            threading.Thread(target=load_image, daemon=True).start()

        except Exception as e:
            logging.error(f"Error initiating WSI load: {str(e)}")
            messagebox.showerror("Error", f"Failed to load Slide: {str(e)}")

    def finalize_image_loading(self, img, png_path):
        """Finalize image loading in the main thread"""
        try:
            # Convert PIL image to PhotoImage
            self.current_image = ImageTk.PhotoImage(img)
            
            # Clear canvas including loading indicator
            self.canvas.delete("all")
            self.clear_current()
            
            # Configure canvas and add image
            self.canvas.config(scrollregion=(0, 0, img.width, img.height))
            self.canvas_image = self.canvas.create_image(
                0, 0,
                image=self.current_image,
                anchor="nw",
                tags='background'
            )
            
            # Store dimensions
            self.original_width = img.width
            self.original_height = img.height
            
            # Update UI elements
            self.current_wsi_label.config(text=f"Current WSI: {self.current_wsi_name}")
            self.status_var.set(f"Loaded: {self.current_wsi_name}")
            
            # Load annotations if available
            self.try_load_annotations(self.current_wsi_name)
            
            # Update viewport
            self.schedule_viewport_update()
            
            # Update default inflammation label
            if hasattr(self, 'default_inflammation_label'):
                self.default_inflammation_label.config(
                    text=f"Default: {self.default_inflammation_status.capitalize()}"
                )
            
        except Exception as e:
            logging.error(f"Error finalizing image load: {str(e)}")
            self.handle_loading_error(str(e))

    def handle_loading_error(self, error_message):
        """Handle image loading errors"""
        self.canvas.delete("all")
        self.status_var.set("Error loading image")
        messagebox.showerror("Error", f"Failed to load Slide: {error_message}")
        
    def try_load_annotations(self, wsi_name: str):
        """Try to load existing annotations for the Slide with scale adjustment"""
        annotation_path = self.annotation_folder / f"{wsi_name}_annotations.json"
        if annotation_path.exists():
            try:
                with open(annotation_path, 'r') as f:
                    loaded_annotations = json.load(f)
                    self.annotations = []
                    for annotation in loaded_annotations:
                        # Deep copy the annotation
                        new_annotation = {
                            "type": annotation["type"],
                            "id": annotation["id"],
                            "properties": annotation["properties"].copy(),
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    # Convert from full resolution to display resolution with scale
                                    [x / self.DOWNSAMPLE * self.current_scale, 
                                    y / self.DOWNSAMPLE * self.current_scale]
                                    for x, y in annotation["geometry"]["coordinates"][0]
                                ]]
                            }
                        }
                        self.annotations.append(new_annotation)

                self.draw_saved_annotations()
                self.update_annotation_list()
                self.status_var.set(f"Loaded: {wsi_name} with existing annotations")
                # Update the WSI list counter
                self.wsi_list.update_annotation_count(wsi_name)

                self.load_clusters()
            except Exception as e:
                logging.error(f"Failed to load annotations for {wsi_name}: {str(e)}")
        
    def load_clusters(self):
        """Load and draw clusters for current WSI"""
        if not self.current_wsi_name:
            return
            
        try:
            cluster_file = self.annotation_folder / "clusters" / f"{self.current_wsi_name}_clusters.json"
            if cluster_file.exists():
                with open(cluster_file, 'r') as f:
                    self.clusters = json.load(f)
                    
                # Draw clusters at correct display coordinates
                for cluster in self.clusters:
                    bounds = cluster["bounds"]
                    self.canvas.create_rectangle(
                        bounds["left"] / self.DOWNSAMPLE * self.current_scale,
                        bounds["top"] / self.DOWNSAMPLE * self.current_scale,
                        bounds["right"] / self.DOWNSAMPLE * self.current_scale,
                        bounds["bottom"] / self.DOWNSAMPLE * self.current_scale,
                        outline='yellow',
                        width=2,
                        tags='cluster_bounds'
                    )
        except Exception as e:
            logging.error(f"Error loading clusters: {e}")

    def delete_all_clusters(self):
        """Delete all clusters for current WSI."""
        if not self.current_wsi_name:
            messagebox.showwarning("Warning", "No WSI loaded")
            return

        if not self.clusters:
            messagebox.showinfo("Info", "No clusters to delete")
            return

        result = messagebox.askyesno(
            "Confirm Delete",
            "Are you sure you want to delete all clusters?"
        )

        if result:
            # Clear clusters list
            self.clusters = []

            # Delete all cluster-related items from the canvas
            self.canvas.delete('cluster_bounds')
            self.canvas.delete('cluster_marker')

            # Reset any scaling or offsets
            self.cluster_start = None
            self.canvas.xview_moveto(0)  # Reset horizontal scroll
            self.canvas.yview_moveto(0)  # Reset vertical scroll

            # Delete the clusters file if it exists
            cluster_file = self.annotation_folder / "clusters" / f"{self.current_wsi_name}_clusters.json"
            if cluster_file.exists():
                cluster_file.unlink()
                
            # Reset cluster counter - annotations with cluster IDs will be cleared
            # Cluster IDs will now start from 1 again for new clusters
            
            # Also update annotations to clear their cluster IDs
            for annotation in self.annotations:
                if "cluster_id" in annotation["properties"]:
                    annotation["properties"]["cluster_id"] = None
            
            # Save annotations to persist the cluster ID removal
            self.save_annotations()
            
            # Update the annotation list
            self.update_annotation_list()

            # Provide user feedback
            self.status_var.set("All clusters deleted - numbering will restart from 1")
            logging.info("All clusters have been deleted successfully.")


    def toggle_clusters_visibility(self):
        """Toggle visibility of cluster rectangles"""
        if not self.current_wsi_name:
            return
            
        self.show_clusters = not self.show_clusters
        
        if self.show_clusters:
            # Redraw clusters
            self.load_clusters()
            self.toggle_clusters_button.config(text="Hide Clusters")
        else:
            # Hide clusters
            self.canvas.delete('cluster_bounds')
            self.toggle_clusters_button.config(text="Show Clusters")

    def save_annotations(self):
        """Save annotations with WSI name and update counter, adjusting for scale"""
        if not self.current_wsi_name:
            messagebox.showwarning("Warning", "No slide loaded.")
            return
        
        # Create annotations directory if needed
        self.annotation_folder.mkdir(parents=True, exist_ok=True)

        file_path = self.annotation_folder / f"{self.current_wsi_name}_annotations.json"
        
        # Allow saving empty list of annotations
        annotations_to_save = []
        if self.annotations:  # Only process if there are annotations
            for annotation in self.annotations:
                # Deep copy the annotation
                new_annotation = {
                    "type": annotation["type"],
                    "id": annotation["id"],
                    "properties": annotation["properties"].copy(),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            # Convert from display coordinates back to full resolution and cast to int
                            [int(x * self.DOWNSAMPLE / self.current_scale), 
                            int(y * self.DOWNSAMPLE / self.current_scale)]
                            for x, y in annotation["geometry"]["coordinates"][0]
                        ]]
                    }
                }
                annotations_to_save.append(new_annotation)
        
        # Save with proper path
        try:
            with open(file_path, 'w') as f:
                json.dump(annotations_to_save, f, indent=2)
                    
            # Force cache clearing for annotation count
            if hasattr(self, 'wsi_list'):
                self.wsi_list.update_annotation_count(self.current_wsi_name)
                    
            self.status_var.set(f"Saved annotations for: {self.current_wsi_name}")
                    
        except Exception as e:
            logging.error(f"Error saving annotations: {str(e)}")
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")

    def save_current_annotation(self, tissue_type: str, inflammation_status: str):
        """Save the current annotation"""
        annotation = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[list(map(int, point)) for point in self.current_points]]  # Convert to integers
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "tissue_type": tissue_type,
                    "inflammation_status": inflammation_status,
                    "color": self.ANNOTATION_OPTIONS['tissue'][tissue_type]  # Add color based on tissue type
                }
            }
        }
        
        # Add to annotations list
        self.annotations.append(annotation)
        
        # Clear current drawing
        self.clear_current()
        
        # Update displays
        self.draw_saved_annotations()
        self.update_annotation_list()
        
        # Autosave to file
        self.save_annotations()
        self.status_var.set(f"New annotation saved: {tissue_type} - {inflammation_status}")
        
        # Ensure the background image is still visible
        if self.current_image and not self.canvas.find_withtag('background'):
            self.canvas.create_image(
                0, 0,
                image=self.current_image,
                anchor="nw",
                tags='background'
            )
            # Make sure background is at the bottom of the stack
            self.canvas.tag_lower('background')

    def draw_saved_annotations(self):
        """Redraw all saved annotations"""
        self.canvas.delete('saved')
        
        # Batch drawing for better performance
        drawing_commands = []
        for annotation in self.annotations:
            coords = annotation['geometry']['coordinates'][0]
            tissue_type = annotation['properties']['classification']['tissue_type']
            
            # Get color from annotation if available, or from options
            if 'color' in annotation['properties']['classification']:
                color = annotation['properties']['classification']['color']
            else:
                color = self.ANNOTATION_OPTIONS['tissue'][tissue_type]
                
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            # Create list of drawing commands
            for i in range(len(coords)):
                p1 = coords[i]
                p2 = coords[(i + 1) % len(coords)]  # Wrap around to first point
                drawing_commands.append((
                    'line',
                    p1[0], p1[1],
                    p2[0], p2[1],
                    hex_color
                ))
        
        # Execute drawing commands in batch
        for cmd in drawing_commands:
            if cmd[0] == 'line':
                self.canvas.create_line(
                    cmd[1], cmd[2], cmd[3], cmd[4],
                    fill=cmd[5],
                    width=3,  # Normal border width
                    tags='saved'
                )
            
        # Redraw highlights if there's a selected annotation
        if hasattr(self, 'selected_annotation'):
            self.highlight_selected_annotation(self.selected_annotation)

    def highlight_selected_annotation(self, annotation, clear_previous=True):
        """Highlight the selected annotation with thicker borders but maintain color"""
        if clear_previous:
            self.canvas.delete('highlight')
            
        try:
            coords = annotation['geometry']['coordinates'][0]
            # Get the tissue type's color
            tissue_type = annotation['properties']['classification']['tissue_type']
            
            # Get color from annotation if available, or from options
            if 'color' in annotation['properties']['classification']:
                color = annotation['properties']['classification']['color']
            else:
                color = self.ANNOTATION_OPTIONS['tissue'][tissue_type]
                
            hex_color = f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}'
            
            # Draw highlighted outline with thicker border of same color
            for i in range(len(coords)):
                p1 = coords[i]
                p2 = coords[(i + 1) % len(coords)]
                self.canvas.create_line(
                    p1[0], p1[1],
                    p2[0], p2[1],
                    fill=hex_color,
                    width=5,  # Thicker border for selected annotation
                    tags='highlight'
                )
            
            # Add center point indicator
            center_x = sum(x for x, y in coords) / len(coords)
            center_y = sum(y for x, y in coords) / len(coords)
            indicator_size = 4
            self.canvas.create_oval(
                center_x - indicator_size,
                center_y - indicator_size,
                center_x + indicator_size,
                center_y + indicator_size,
                fill='yellow',
                outline='black',
                tags='highlight'
            )
            
        except Exception as e:
            logging.error(f"Error highlighting annotation: {str(e)}")
    
    def update_annotation(self, annotation, tissue_type: str, inflammation_status: str):
        """Update an existing annotation"""
        # Update the color based on the tissue type
        color = self.ANNOTATION_OPTIONS['tissue'][tissue_type]
        
        annotation['properties']['classification'].update({
            'tissue_type': tissue_type,
            'inflammation_status': inflammation_status,
            'color': color
        })
        
        # Redraw annotations
        self.draw_saved_annotations()
        
        # Clear selection highlight
        self.canvas.delete('highlight')
        if hasattr(self, 'selected_annotation'):
            delattr(self, 'selected_annotation')
        
        # Autosave
        self.save_annotations()
        
        # Update the annotation list
        self.update_annotation_list()
    
    def clear_current(self):
        """Clear only the current drawing without affecting the background"""
        self.canvas.delete('current')  # Only delete items tagged as 'current'
        self.current_points = []
        self.drawing = False
        
        # Check if the background image is still visible
        if self.current_image and not self.canvas.find_withtag('background'):
            # Redraw the background image if it's missing
            self.canvas_image = self.canvas.create_image(
                0, 0,
                image=self.current_image,
                anchor="nw",
                tags='background'
            )
            # Make sure it's at the back
            self.canvas.tag_lower('background')
        
    def show_annotation_dialog(self, existing_annotation=None):
        """Show dialog for annotation details"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Annotation Details")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Store dialog state
        self.current_dialog = dialog
        self.dialog_saved = False
        
        # Center dialog
        window_width = 400
        window_height = 250
        position_x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (window_width // 2)
        position_y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (window_height // 2)
        dialog.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Tissue selection
        tissue_frame = ttk.LabelFrame(dialog, text="Tissue Type")
        tissue_frame.pack(padx=10, pady=5, fill=tk.X)
        
        tissue_options = list(self.ANNOTATION_OPTIONS['tissue'].keys())
        selected_tissue = tk.StringVar(value=tissue_options[0])
        ttk.Label(tissue_frame, text="Tissue:").pack(side=tk.LEFT, padx=5)
        tissue_menu = ttk.OptionMenu(tissue_frame, selected_tissue, 
                    tissue_options[0], *tissue_options)
        tissue_menu.pack(side=tk.LEFT, padx=5)
        
        # Inflammation selection
        inflammation_frame = ttk.LabelFrame(dialog, text="Inflammation Status")
        inflammation_frame.pack(padx=10, pady=5, fill=tk.X)
        
        inflammation_options = list(self.ANNOTATION_OPTIONS['inflammation_status'].keys())
        selected_inflammation = tk.StringVar(value=self.default_inflammation_status)
        ttk.Label(inflammation_frame, text="Inflammation:").pack(side=tk.LEFT, padx=5)
        inflammation_menu = ttk.OptionMenu(inflammation_frame, selected_inflammation,
                        self.default_inflammation_status, *inflammation_options)
        inflammation_menu.pack(side=tk.LEFT, padx=5)
        
        # Get values from existing annotation if editing
        if existing_annotation:
            props = existing_annotation['properties']['classification']
            selected_tissue.set(props.get('tissue_type', tissue_options[0]))
            # Handle both old and new naming convention for backward compatibility
            inflammation_status = props.get('inflammation_status', 
                                props.get('inflammation', self.default_inflammation_status))
            selected_inflammation.set(inflammation_status)
        
        def save_and_close(event=None):
            """Save annotation and close dialog"""
            tissue_type = selected_tissue.get()
            inflammation_status = selected_inflammation.get()
            
            # Save as defaults if checkbox is checked
            if save_defaults_var.get():
                # Update default inflammation status for future annotations
                self.default_inflammation_status = inflammation_status
                # Also save the tissue type as default
                self.last_tissue_type = tissue_type
                
                # Update UI if needed
                if hasattr(self, 'default_inflammation_label'):
                    self.default_inflammation_label.config(
                        text=f"Default: {self.default_inflammation_status.capitalize()}"
                    )
            
            if existing_annotation:
                self.update_annotation(existing_annotation, tissue_type, inflammation_status)
            else:
                self.save_current_annotation(tissue_type, inflammation_status)
            
            self.dialog_saved = True
            dialog.destroy()
        
        def on_dialog_close():
            """Handle dialog closing without saving"""
            if not self.dialog_saved and not existing_annotation:
                self.clear_current()
            dialog.destroy()
        
        # Save as defaults checkbox
        save_defaults_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            dialog, 
            text="Use as default for future annotations", 
            variable=save_defaults_var
        ).pack(pady=5, padx=10, anchor=tk.W)
        
        # Create a frame to hold the buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        # Create Cancel button on the left
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_dialog_close)
        cancel_button.pack(side='left', padx=5)

        # Create Save button on the right
        save_button = ttk.Button(button_frame, text="Save", command=save_and_close)
        save_button.pack(side='left', padx=5)
        
        # Set default button (highlighted when Enter is pressed)
        save_button.focus_set()
        
        # Keyboard shortcuts
        dialog.bind("<Return>", save_and_close)  # Enter to save
        dialog.bind("<Escape>", lambda e: on_dialog_close())  # Escape to cancel
        
        # Bind closing event
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
        
        # Set focus to the dialog
        dialog.focus_set()
        
    def start_drawing(self, event):
        """Start drawing a polygon"""
        if not self.current_image:
            return
        
        # Get canvas coordinates considering scroll position
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.drawing = True
        self.current_points = [(canvas_x, canvas_y)]
        self.canvas.create_oval(
            canvas_x-2, canvas_y-2,
            canvas_x+2, canvas_y+2,
            fill='red',  # Change to red for better visibility
            tags='current'
        )
        
        # Update status
        self.status_var.set("Drawing annotation - click and drag to continue, release to finish")
    
    def continue_drawing(self, event):
        """Continue drawing the polygon"""
        if not self.drawing:
            return
        
        # Get canvas coordinates considering scroll position
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        self.current_points.append((canvas_x, canvas_y))
        if len(self.current_points) > 1:
            p1 = self.current_points[-2]
            p2 = (canvas_x, canvas_y)
            self.canvas.create_line(
                p1[0], p1[1],
                p2[0], p2[1],
                fill='red',  # Change to red for better visibility
                width=2,
                tags='current'
            )

    def end_move(self, event):
        """End annotation movement and save changes"""
        if hasattr(self, 'selected_annotation'):
            try:
                # Reset cursor
                self.canvas.config(cursor="")
                
                # Save changes
                self.save_annotations()
                
                # Update annotation list to reflect any changes
                self.update_annotation_list()
                
                # Clean up movement variables
                if hasattr(self, 'move_start_x'):
                    delattr(self, 'move_start_x')
                if hasattr(self, 'move_start_y'):
                    delattr(self, 'move_start_y')
                if hasattr(self, 'original_coords'):
                    delattr(self, 'original_coords')
                
                self.status_var.set("Annotation moved and saved")
                
            except Exception as e:
                logging.error(f"Error ending move: {str(e)}")
                self.status_var.set("Error saving moved annotation")
    
    def end_drawing(self, event):
        """End drawing and prompt for annotation details"""
        if not self.drawing or len(self.current_points) < 3:
            self.clear_current()
            self.drawing = False  # Ensure drawing state is reset
            return
        
        self.drawing = False
        
        # Close the polygon
        first = self.current_points[0]
        last = self.current_points[-1]
        self.canvas.create_line(
            last[0], last[1],
            first[0], first[1],
            fill='red',  # Change to red for better visibility
            width=2,
            tags='current'
        )
        
        # Show annotation dialog
        self.show_annotation_dialog()
        
        # Ensure the background image is still visible if dialog is cancelled
        if self.current_image and not self.canvas.find_withtag('background'):
            self.canvas.create_image(
                0, 0,
                image=self.current_image,
                anchor="nw",
                tags='background'
            )
            # Make sure background is at the bottom of the stack
            self.canvas.tag_lower('background')
    
    def run(self):
        """Start the application"""
        self.root.mainloop()
if __name__ == "__main__":
    app = AnnotationTool()
    app.run()