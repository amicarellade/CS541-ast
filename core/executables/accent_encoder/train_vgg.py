import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision

import hparams as hp
from datasets import *
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder

# File paths
METADATA_PATH = "../Vctk-Corpus/mel_spec_metadata.json"
MODEL_SAVE_PATH = f"{hp.models_save_path}{hp.f_delim}acc_encoder{hp.f_delim}vgg{hp.f_delim}vgg_acc"
RESULTS_LOG_FILE = f"{hp.models_save_path}{hp.f_delim}acc_encoder{hp.f_delim}vgg{hp.f_delim}training_results.txt"

# Hyperparameter search space
batch_sizes = [32,64]
learning_rates = [0.001, 0.0001]
loss_functions = [nn.CrossEntropyLoss, nn.NLLLoss]
schedulers = ["StepLR", "ReduceLROnPlateau","None"]
optimizers = ["Adam", "SGD"]

# Training configurations
NUM_EPOCHS = 10

# Define a function to initialize an optimizer
def get_optimizer(name, params, lr):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def preprocess_df(df: pd.DataFrame):

    # Encodings for accents, OHE vector will be supplied in dataset
    encoder = LabelEncoder()
    accents = encoder.fit(df["accents"])
    df["enc_accents"] = accents

    return df


# Train the model
def train_model(config):
    batch_size, lr, loss_func, scheduler_name, optimizer_name = config

    print(f"\nStarting training with config: Batch Size={batch_size}, LR={lr}, Loss={loss_func.__name__}, Scheduler={scheduler_name}, Optimizer={optimizer_name}")

    # Create DataLoaders
    train_loader, val_loader, num_classes = create_dataloaders(METADATA_PATH, batch_size=batch_size)

    # Load pre-trained VGG19 model
    vgg = torchvision.models.vgg19(pretrained=True)

    # Modify the classifier to match the number of accent classes
    vgg.classifier[6] = nn.Linear(4096, num_classes)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg.to(device)

    # Define loss function and optimizer
    criterion = loss_func()
    optimizer = get_optimizer(optimizer_name, vgg.parameters(), lr)

    # Define learning rate scheduler
    scheduler = None  # Initialize scheduler to None
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)
    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Training phase
        vgg.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = vgg(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_pbar.set_postfix(loss=loss.item())

        train_accuracy = 100 * correct / total
        print(f"Training Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        vgg.eval()
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc="Validating", leave=False)
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = vgg(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")

        # Adjust learning rate if using ReduceLROnPlateau
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_loss / len(val_loader))
        elif scheduler_name == "StepLR":
            scheduler.step()

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = f"{MODEL_SAVE_PATH}_batch{batch_size}_lr{lr}_{loss_func.__name__}_{scheduler_name}_{optimizer_name}.pth"
            torch.save(vgg.state_dict(), model_path)
            print(f"Best model saved to {model_path} with Validation Accuracy: {best_val_accuracy:.2f}%")

    # Log results
    with open(RESULTS_LOG_FILE, "a") as f:
        f.write(f"Config: Batch Size={batch_size}, LR={lr}, Loss={loss_func.__name__}, Scheduler={scheduler_name}, Optimizer={optimizer_name}\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy:.2f}%\n\n")

    print(f"Training complete for config: Batch Size={batch_size}, LR={lr}, Loss={loss_func.__name__}, Scheduler={scheduler_name}, Optimizer={optimizer_name}\n")




if __name__ == "__main__":

    df = pd.read_csv(os.path.join(hp.data_path, hp.vctk_csv))

    df = preprocess_df(df)


    pass