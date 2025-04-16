# MNIST Classification with PyTorch

This project demonstrates how to build and train a convolutional neural network (CNN) on the MNIST dataset using raw data files. The repository contains modular Python scripts for loading data, defining the model, training, evaluating, and plotting training progress.

## Project Structure

```bash
MNIST/
├── data_loader.py       # Module to load raw MNIST data files and preprocess them.
├── model.py             # Definition of the CNN model using PyTorch.
├── train.py             # Main script for training, evaluating, and plotting results.
├── requirements.txt     # Python dependencies for the project.
├── README.md            # Project overview and setup instructions.
└── mnist/
    └── MNIST/
        └── raw/         # Directory containing raw MNIST files.
            ├── t10k-images-idx3-ubyte
            ├── t10k-labels-idx1-ubyte
            ├── train-images-idx3-ubyte
            ├── train-labels-idx1-ubyte
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte.gz
            └── train-labels-idx1-ubyte.gz
```

- **data_loader.py:**  
  Contains helper functions to read raw MNIST files (such as `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`), normalize the image data, and format it for training.

- **model.py:**  
  Implements a simple convolutional neural network (CNN) designed for classifying MNIST digits. The network includes two convolutional layers followed by pooling and two fully-connected layers.

- **train.py:**  
  Serves as the project entry point. It loads the data using `data_loader.py`, trains the CNN model defined in `model.py`, evaluates its performance on the test set, and plots training loss and test accuracy over epochs using matplotlib.

- **requirements.txt:**  
  Contains the dependencies needed to run the project. This includes NumPy, PyTorch, and Matplotlib.

## Installation

1. **Clone the repository or copy the project files** to your local machine.

2. **Place the raw MNIST files** in the data directory. The expected raw file names include:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`

   By default, the project expects these files to be located at:
`/Users/kennyyu/Desktop/projects/AI/MNIST/mnist/MNIST/raw`
Adjust the path if your files are located elsewhere.

3. **Install the dependencies:**  
Ensure you have Python 3 installed, then run:
```bash
pip install -r requirements.txt
```

## Usage
To train and evaluate the model along with generating plots for training loss and test accuracy, run the training script from the command line:
```bash
python train.py --data_dir "/Users/kennyyu/Desktop/projects/AI/MNIST/mnist/MNIST/raw" --epochs 5 
                --batch_size 64 
                --lr 0.001
```
### Command-Line Arguments:
- `--data_dir`: Path to the directory containing the raw MNIST data files. Adjust the path according to your desired directory.
- `--epochs`: Batch size for training (default: 64).
- `--lr`: Learning rate (default: 0.001).

## Output
- **Console Output:** Displays training progress per epoch and test performance statistics (loss and accuracy).
- **Plots:** After training, two plots will be generated and saved:
  - `training_loss.png`: Training loss per epoch.
  - `test_accuracy.png`: Test accuracy per epoch.
- **Model Checkpoint:** The trained model is saved as `mnist_model.pth` in the project directory.
