#!/usr/bin/env python3
"""
GSDB Project Setup Script

This script:
1. Creates necessary project directories
2. Sets up the conda environment
3. Installs the project as a package
4. Downloads sample data (optional)
5. Verifies the installation

Usage:
    python setup_project.py [--data_path /path/to/data] [--download_samples] [--skip_conda]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import logging
import shutil
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def run_command(command, description=None, check=True):
    """Run a shell command with proper logging and error handling."""
    if description:
        logging.info(description)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {e}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise

def create_directories(base_path):
    """Create the required project directory structure."""
    logging.info("Creating project directories...")
    
    # Define directories to create
    directories = [
        "data/raw/slides",
        "data/raw/annotations",
        "data/processed/tiles",
        "data/processed/downsized_slides",
        "data/splits/seed_42",
        "results/logs",
        "results/figures",
        "results/tables",
        "results/models",
        "results/metrics",
        "results/evaluations",
        "results/tuning"
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created: {dir_path}")
    
    return True

def setup_conda_environment():
    """Setup conda environment from environment.yml file."""
    logging.info("Setting up conda environment...")
    
    # Check if conda is available
    try:
        run_command("conda --version", "Checking conda installation")
    except:
        logging.error("Conda not found. Please install Miniconda or Anaconda first.")
        return False
    
    # Create conda environment from environment.yml
    try:
        run_command("conda env create -f environment.yml", "Creating conda environment from environment.yml")
        logging.info("Conda environment 'gsdb' created successfully.")
        
        # Provide activation instructions
        logging.info("\nTo activate the environment, run:")
        logging.info("    conda activate gsdb")
        return True
    except Exception as e:
        logging.error(f"Failed to create conda environment: {e}")
        return False


def download_sample_data(data_path):
    """Download sample data for testing the installation."""
    logging.info("Downloading sample data...")
    
    sample_data_url = "https://huggingface.co/datasets/DominicLiebel/GSDB/resolve/main/sample_slides.zip"
    sample_data_zip = data_path / "sample_slides.zip"
    sample_data_dir = data_path / "raw"
    
    # Create a simple progress indicator
    def progress_callback(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\rDownloading: {percent}% complete")
        sys.stdout.flush()
    
    try:
        import urllib.request
        
        # Download the data
        logging.info(f"Downloading from {sample_data_url}...")
        urllib.request.urlretrieve(
            sample_data_url, 
            sample_data_zip,
            reporthook=progress_callback
        )
        print()  # Newline after progress indicator
        
        # Extract the data
        logging.info("Extracting sample data...")
        import zipfile
        with zipfile.ZipFile(sample_data_zip, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # Move files to correct locations
        extract_dir = data_path / "GSDB-sample-data-main"
        for src_dir in extract_dir.glob("*"):
            if src_dir.is_dir():
                dest_dir = sample_data_dir / src_dir.name
                if dest_dir.exists():
                    # Merge contents
                    for item in src_dir.glob("*"):
                        shutil.move(str(item), str(dest_dir))
                else:
                    # Move the whole directory
                    shutil.move(str(src_dir), str(dest_dir))
        
        # Clean up
        if sample_data_zip.exists():
            sample_data_zip.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
            
        logging.info("Sample data downloaded and extracted successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to download sample data: {e}")
        return False

def verify_installation(data_path):
    """Verify the installation by running a simple test."""
    logging.info("Verifying installation...")
    
    # Set the environment variable for the base directory
    os.environ["GASTRIC_BASE_DIR"] = str(data_path)
    
    # Run a simple script that uses the library
    verification_script = """
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.config.paths import get_project_paths
    paths = get_project_paths()
    print("\\nGSDB paths successfully initialized:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    print("\\nVerification successful!")
    sys.exit(0)
except Exception as e:
    print(f"\\nVerification failed: {e}")
    sys.exit(1)
"""
    
    # Write the verification script to a temporary file
    verification_file = Path("verify_install.py")
    with open(verification_file, "w") as f:
        f.write(verification_script)
    
    try:
        # Run the verification script
        result = run_command(f"python {verification_file}", "Running verification script")
        if "Verification successful!" in result.stdout:
            logging.info("Installation verified successfully!")
            return True
        else:
            logging.error("Verification failed.")
            logging.error(result.stdout)
            logging.error(result.stderr)
            return False
    finally:
        # Clean up the temporary file
        if verification_file.exists():
            verification_file.unlink()

def main():
    parser = argparse.ArgumentParser(description="GSDB Project Setup Script")
    parser.add_argument("--data_path", type=Path, default=Path.cwd(),
                        help="Base directory for data and results")
    parser.add_argument("--download_samples", action="store_true",
                        help="Download sample data for testing")
    parser.add_argument("--skip_conda", action="store_true",
                        help="Skip conda environment setup")
    
    args = parser.parse_args()
    
    logging.info("Starting GSDB project setup")
    logging.info(f"Data path: {args.data_path}")
    
    start_time = time.time()
    
    # Create directories
    if not create_directories(args.data_path):
        logging.error("Failed to create directories")
        return 1
    
    # Setup conda environment (unless skipped)
    if not args.skip_conda:
        if not setup_conda_environment():
            logging.error("Failed to setup conda environment")
            return 1
    else:
        logging.info("Skipping conda environment setup (--skip_conda)")
    
    # Download sample data (if requested)
    if args.download_samples:
        if not download_sample_data(args.data_path):
            logging.warning("Failed to download sample data")
    
    # Verify installation
    if not verify_installation(args.data_path):
        logging.error("Failed to verify installation")
        return 1
    
    # Calculate setup time
    elapsed_time = time.time() - start_time
    logging.info(f"Setup completed in {elapsed_time:.1f} seconds")
    
    # Final instructions
    logging.info("\n" + "="*60)
    logging.info("GSDB setup completed successfully!")
    logging.info("\nTo activate the environment:")
    logging.info("    conda activate gsdb")
    logging.info("\nTo start using the project, set the environment variable:")
    logging.info(f"    export GASTRIC_BASE_DIR=\"{args.data_path}\"")
    logging.info("\nTo run the pipeline:")
    logging.info("    python src/data/process_dataset.py")
    logging.info("    python src/models/train.py --task inflammation --model convnext_large")
    logging.info("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())