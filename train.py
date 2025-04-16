import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Import our modules
from data_loader import load_dataset
from model import MNISTNet


def get_data_loaders(data_dir, batch_size=64):
    # Load raw data using the helper function
    (train_images, train_labels), (test_images, test_labels) = load_dataset(data_dir)

    # Convert numpy arrays to Torch tensors
    train_images = torch.tensor(train_images)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(test_images)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create datasets and corresponding data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch, loss_history):
    model.train()
    running_loss = 0.0
    batch_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # zero the parameter gradients
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # backpropagation
        optimizer.step()  # update weights

        running_loss += loss.item()
        batch_count += 1
        # Print progress for every 100 batches (optional)
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    # Append average loss for this epoch
    avg_loss = running_loss / batch_count if batch_count else 0.0
    loss_history.append(avg_loss)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='MNIST Training with Plotting')
    parser.add_argument('--data_dir', type=str, default="/Users/kennyyu/Desktop/projects/AI/MNIST/mnist/MNIST/raw",
                        help='Directory containing raw MNIST files')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    # Initialize model, loss function, and optimiser
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    # Lists for storing loss and accuracy history
    train_loss_history = []
    test_accuracy_history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimiser, criterion, epoch, train_loss_history)
        accuracy = test(model, device, test_loader, criterion)
        test_accuracy_history.append(accuracy)

    # Plot Training Loss History
    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_loss_history, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()

    # Plot Test Accuracy History
    plt.figure()
    plt.plot(range(1, args.epochs + 1), test_accuracy_history, marker='o')
    plt.title("Test Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig("test_accuracy.png")
    plt.show()

    # Optionally, save the trained model.
    model_path = os.path.join("mnist_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()

