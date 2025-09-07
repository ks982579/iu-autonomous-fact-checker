"""
Script to compress a trained model directory into a tarball,
excluding heavy checkpoint directories.

Current implementation - Run in directory THIS file live in
didn't compress much at all...
"""

import tarfile
import os
from pathlib import Path
from datetime import datetime

def should_exclude(tarinfo):
    """
    Filter function to exclude checkpoint directories and other unwanted files
    """
    # Exclude checkpoint directories
    if 'checkpoint-' in tarinfo.name:
        print(f"Excluding: {tarinfo.name}")
        return None
    
    # Exclude other potentially large/unnecessary files
    exclude_patterns = [
        'runs/',           # TensorBoard logs
        '__pycache__/',    # Python cache
        '.git/',           # Git directory
        'logs/',           # Training logs (optional)
        '.DS_Store',       # macOS files
    ]
    
    for pattern in exclude_patterns:
        if pattern in tarinfo.name:
            print(f"Excluding: {tarinfo.name}")
            return None
    
    return tarinfo

def compress_model(model_dir="./claim-extractor/trainingresults/latest", output_name=None):
    """
    Compress the model directory into a tarball
    
    Args:
        model_dir: Path to the model directory
        output_name: Name for the output tarball (optional)
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"Error: Model directory '{model_path}' does not exist!")
        return
    
    # Generate output filename if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"./claim-extractor/bert_claim_model_{timestamp}.tar.gz"
    
    print(f"Compressing model from: {model_path}")
    print(f"Output file: {output_name}")
    print(f"Excluding checkpoint-* directories...\n")
    
    # Create compressed tarball
    with tarfile.open(output_name, "w:gz") as tar:
        # Add the directory with filtering
        tar.add(model_path, arcname=model_path.name, filter=should_exclude)
    
    # Show results
    original_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    compressed_size = Path(output_name).stat().st_size
    
    print(f"\nCompression complete!")
    print(f"Original size: {original_size / (1024**2):.1f} MB")
    print(f"Compressed size: {compressed_size / (1024**2):.1f} MB")
    print(f"Compression ratio: {compressed_size / original_size:.2%}")
    print(f"Saved: {output_name}")

def list_model_contents(model_dir="./claim-extractor/trainingresults/latest"):
    """
    Show what files are in the model directory
    """
    model_path = Path(model_dir)
    print(f"Contents of {model_path}:")
    print("-" * 40)
    
    total_size = 0
    for item in sorted(model_path.rglob('*')):
        if item.is_file():
            size_mb = item.stat().st_size / (1024**2)
            total_size += size_mb
            print(f"{size_mb:6.1f} MB  {item.relative_to(model_path)}")
        elif item.is_dir() and 'checkpoint-' in item.name:
            print(f"  [DIR]   {item.relative_to(model_path)} (will be excluded)")
    
    print("-" * 40)
    print(f"Total size: {total_size:.1f} MB")

if __name__ == "__main__":
    # show what is in directory
    list_model_contents()
    print("\n" + "-"*50 + "\n")
    compress_model()

# ---------------

def extract_model(tarball_path, extract_to="."):
    """
    Extract the model tarball

    Or just run `tar -xzf -v <tar-ball> -C <destination-path>
    """
    print(f"Extracting {tarball_path} to {extract_to}")
    
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    
    print("Extraction complete!")