import argparse
import yaml
import os
import torch
from glob import glob
from PIL import Image
from .pipeline import HumanCrafterPipeline
from .models import HumanCrafterModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace
    from types import SimpleNamespace
    config = SimpleNamespace(**config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = HumanCrafterModel(config.model)
    if hasattr(config.model, "checkpoint_path") and config.model.checkpoint_path:
        model.load_state_dict(torch.load(config.model.checkpoint_path))
    
    # Create pipeline
    pipeline = HumanCrafterPipeline(model)
    
    # Get input images
    image_paths = glob(os.path.join(args.input_dir, "*.jpg")) + glob(os.path.join(args.input_dir, "*.png"))
    
    # Process each image
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        
        # Load image
        image = Image.open(image_path)
        
        # Run inference
        results = pipeline(image)
        
        # Save results
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_subdir = os.path.join(args.output_dir, image_name)
        pipeline.save_results(results, output_subdir)
    
    print(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()