#!/usr/bin/env python3

# Assignment 5
# CMPU 366, Fall 2025

import csv
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

torch.manual_seed(0)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)


EPOCHS = 5
BATCH_SIZE = 24
MAX_LENGTH = 10
LR = 0.001
CKPT_DIR = "./ckpt"
NUM_CLASSES = 3


label_map = {"Beyoncé": 0, "Drake": 1, "Taylor Swift": 2}
label_map_rev = {0: "Beyoncé", 1: "Drake", 2: "Taylor Swift"}


class NN(nn.Module):
    def __init__(self, n_features: int):
        """Construct the pieces of the neural network."""

        # Initialize the parent class (nn.Module)
        super(NN, self).__init__()

        # Add the pretrained DistilBERT model
        self.bert = bert
        
        # Four hidden layers with ReLU activations
        # Each reduces dimensionality by half: 768 → 384 → 192 → 96 → 48
        self.hidden_layers = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU()
        )
        
        # Flatten layer after hidden layers
        self.flatten = nn.Flatten()
        
        # Output layer: (max_length * 48) → 3 classes
        output_input_dim = n_features * 48  # MAX_LENGTH * 48 = 10 * 48 = 480
        self.output_layer = nn.Linear(output_input_dim, 3)

        # Log probabilities of each class
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """The forward pass of the model: Transform the input data x into
        the output predictions (the log probabilities for each label).
        """
        # Pass input token IDs through DistilBERT
        outputs = self.bert(x)
        hidden_states = outputs.last_hidden_state
        # Shape: (batch_size, max_length, 768)
        
        # Pass through hidden layers (operates on each token's embedding)
        hidden = self.hidden_layers(hidden_states)
        # Shape: (batch_size, max_length, 48)
        
        # Flatten to (batch_size, max_length * 48)
        flattened = self.flatten(hidden)
        
        # Pass through output layer and softmax
        scores = self.output_layer(flattened)
        probs = self.out(scores)
        return probs

####


def make_data(fname: str, label_map: dict) -> Tuple[list[str], list[int]]:
    lyrics = []
    labels = []
    
    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            artist_name = row[0]  # First column
            lyric_line = row[3]   # Fourth column (0-indexed)
            
            lyrics.append(lyric_line)
            labels.append(label_map[artist_name])
    
    return lyrics, labels


def prep_bert_data(
    data: list[str], max_length: int
) -> list[torch.Tensor]:
    feats = []
    
    for lyric in data:
        # Tokenize with truncation and padding to max_length
        encoded = tokenizer(
            lyric,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        
        # Extract input_ids (ignoring attention_mask for this assignment)
        input_ids = encoded['input_ids']
        
        # Convert to tensor with dtype=torch.long
        tensor_ids = torch.tensor(input_ids, dtype=torch.long)
        feats.append(tensor_ids)
    
    return feats


####


def get_predicted_label_from_predictions(predictions):
    predicted_label = predictions.argmax(1).item()
    return predicted_label


def sample_and_print_predictions(feats, data, labels, model):
    """Randomly sample and print predictions for 10 lyrics."""
    import random
    
    # Get 10 random indices from the dataset
    num_samples = min(10, len(data))
    indices = random.sample(range(len(data)), num_samples)
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            # Get the lyric, true label, and features
            lyric = data[idx]
            true_label = labels[idx]
            feat = feats[idx].unsqueeze(0).to(device)  # Add batch dimension
            
            # Get model prediction
            pred = model(feat)
            predicted_label = pred.argmax(1).item()
            
            # Convert numeric labels to artist names
            true_artist = label_map_rev[true_label]
            predicted_artist = label_map_rev[predicted_label]
            
            # Print results
            print(f"Lyrics: {lyric}")
            print(f"- Class: {true_artist}")
            print(f"- Prediction: {predicted_artist}")
            print()


def print_performance_by_class(labels, predictions):
    """Print accuracy for each class.
    """
    print("Accuracy by Category:")
    
    for category in range(NUM_CLASSES):
        # Count total instances and correct predictions for this category
        total = 0
        correct = 0
        
        for i, true_label in enumerate(labels):
            if true_label == category:
                total += 1
                # Get predicted label from tensor
                pred_tensor = predictions[i]
                predicted_label = pred_tensor.argmax(1).item()
                
                if predicted_label == category:
                    correct += 1
        
        # Calculate and print accuracy
        if total > 0:
            accuracy = correct / total
            print(f"Category {category}: {accuracy:.1f}")
        else:
            print(f"Category {category}: N/A (no samples)")
    
    print()

####


def train(dataloader, model, optimizer, epoch: int):
    """Run an epoch of training the model on the provided data, using the
    specified optimizer.
    """
    # Calculate class weights based on training data frequencies
    # Class counts from the output:
    # Beyoncé (0): 3580, Drake (1): 4662, Taylor Swift (2): 7238
    class_counts = {0: 3580, 1: 4662, 2: 7238}
    total_samples = sum(class_counts.values())
    
    # Calculate weights: inverse of frequency, normalized
    class_weights = {}
    for label, count in class_counts.items():
        class_weights[label] = total_samples / (NUM_CLASSES * count)
    
    # Convert to tensor in the correct order [class 0, class 1, class 2]
    weight_tensor = torch.tensor([
        class_weights[0],
        class_weights[1],
        class_weights[2]
    ], dtype=torch.float).to(device)
    
    # Pass weights to the loss function
    loss_fn = nn.NLLLoss(weight=weight_tensor)
    
    model.train()
    with tqdm(dataloader, unit="batch") as tbatch:
        for X, y in tbatch:
            X = X.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{CKPT_DIR}/ckpt_{epoch}.pt",
    )

def predict(data, model):
    predictions = []
    dataloader = DataLoader(data, batch_size=1)
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            predictions.append(pred)
    return predictions


def test(dataloader, model, dataset_name):
    loss_fn = nn.NLLLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"{dataset_name} Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>6f}\n"
    )


####


def make_or_restore_model(
    nfeat: int,
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    """Either restore the latest model, or create a fresh one"""
    model = NN(nfeat).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    ckpt_path = Path(CKPT_DIR)
    checkpoints = [p for p in ckpt_path.glob("*.pt")]

    if checkpoints:
        latest_checkpoint = max(
            checkpoints,
            key=lambda p: int(p.stem.split('_')[1])
        )
        print("Restoring from", latest_checkpoint)
        ckpt = torch.load(latest_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt["epoch"]
        return model, optimizer, epoch + 1
    else:
        print("Creating a new model")
        return model, optimizer, 0


####


def main():
    """Run the song classification."""

    Path(CKPT_DIR).mkdir(exist_ok=True)

    train_f = "train.csv"
    test_f = "test.csv"

    train_data, train_labels = make_data(train_f, label_map)
    test_data, test_labels = make_data(test_f, label_map)

    for i in label_map_rev:
        print(f"Lyrics in Class {i} ({label_map_rev[i] + '):':14}",
              len([t for t in train_labels if t == i]))

    print()

    train_feats = prep_bert_data(train_data, MAX_LENGTH)
    test_feats = prep_bert_data(test_data, MAX_LENGTH)

    train_dataset = list(zip(train_feats, train_labels))
    test_dataset = list(zip(test_feats, test_labels))

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # print(train_feats[0])

    model, optimizer, epoch_start = make_or_restore_model(MAX_LENGTH)

    for e in range(epoch_start, EPOCHS):
        print()
        print("Epoch", e)
        print("-------")
    
        model.train()
        train(train_dataloader, model, optimizer, e)
    
        print()
    
        model.eval()
        test(train_dataloader, model, "Train")
        test(test_dataloader, model, "Test")
    
    test_predictions = predict(test_feats, model)
    print_performance_by_class(test_labels, test_predictions)
    print()

    sample_and_print_predictions(test_feats, test_data, test_labels,model)


if __name__ == "__main__":
    main() 
