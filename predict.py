# predict.py

import argparse
import torch
import json
import model_utils
import image_utils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image with a trained model checkpoint.")
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    
    # Load model checkpoint
    model = model_utils.load_checkpoint(args.checkpoint)
    
    # Process input image
    img = image_utils.process_image(args.input)
    
    # Predict top K classes
    probs, classes = model_utils.predict(img, model, args.top_k, args.gpu)
    
    # Load category names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]
    
    # Print prediction results
    for prob, cls in zip(probs, classes):
        print(f'{cls}: {prob*100:.2f}%')

if __name__ == '__main__':
    main()
