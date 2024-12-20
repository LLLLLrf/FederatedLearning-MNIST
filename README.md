# Federated Learning - MNIST

## Requirements

- kaggle
- torch
- opencv-python
- matplotlib
- numpy

## Download Dataset

```bash
kaggle datasets download hojjatk/mnist-dataset
mkdir raw_data
unzip mnist-dataset.zip -d raw_data
```

## Data Preprocess

```bash
python test_dataset.py
python train_dataset.py
```

## Train

Centralized Training

```bash
python train.py
```

Fedrated Learning

```bash
python fedrated_learning.py
```

## Test

```bash
python test.py
```
