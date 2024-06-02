# train.py

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import model_utils
import data_utils

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
    parser.add_argument('data_dir', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    
    # Load data
    trainloader, validloader, _ = data_utils.load_data(args.data_dir)

    # Load model architecture
    model = model_utils.load_model(args.arch, args.hidden_units)
    
    # Train model
    model_utils.train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)
    
    # Save trained model checkpoint
    model_utils.save_checkpoint(model, args.save_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs)

if __name__ == '__main__':
    main()
