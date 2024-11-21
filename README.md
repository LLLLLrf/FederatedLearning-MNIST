# Federated Learning - MNIST

## requirements

- kaggle
- torch
- opencv-python
- matplotlib

## download dataset

```bash
kaggle datasets download hojjatk/mnist-dataset
mkdir raw_data
unzip mnist-dataset.zip -d raw_data
```

## data preprocess

```bash
python test_dataset.py
python train_dataset.py
```

## train

```bash
python train.py
```