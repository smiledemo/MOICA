import argparse
import torch
from train import train_model
from test import test_model


def main():
    parser = argparse.ArgumentParser(description='MOICA Project')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--dataset', type=str, choices=['OfficeHome', 'Office31', 'MNIST', 'SVHN', 'USPS'],
                        required=True, help='Dataset name')
    parser.add_argument('--domain', type=str, required=True, help='Domain name (for OfficeHome and Office31)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model name for feature extractor')

    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)


if __name__ == '__main__':
    main()
