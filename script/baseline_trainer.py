import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
import torch.nn as nn

import baseline_dataset

def train(num_exp_id=0, 
          num_batch_size=64, 
          num_epochs=30, 
          num_test_fold=0,
          model=resnet18,
          weights=ResNet18_Weights.IMAGENET1K_V1,
          lr=1e-5,
          weight_decay=0,
          num_class=3,
          output_dir="../result"
          ):
    print(f"Training baseline model...")
    os.makedirs(f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/weights", exist_ok=True)
    os.makedirs(f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/csv", exist_ok=True)
    
    # setting hyper parameters
    BATCH_SIZE = num_batch_size
    EPOCHS = num_epochs
    
    # augmentations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10)
        ])

    # load data
    train_ds = baseline_dataset.BaselineDataset(
        "..//data/csv/data_divide.csv", 
        transform=transform, 
        mode="train", 
        num_test_fold=num_test_fold
        )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define model
    model = model(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    
    model.to(device)
    
    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # train
    model.train()
    
    train_loss_list = []
    train_acc_list = []
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(train_dl):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        
        train_loss = running_loss / len(train_dl)
        train_acc = correct / total

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        print(f"epoch: {epoch+1}, loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
        
    torch.save(model.state_dict(), f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/weights/baseline.pt")
    
    df = pd.DataFrame({
        "train_loss": train_loss_list,
        "train_acc": train_acc_list
        })
    df.to_csv(f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/csv/baseline_history.csv", index=False)

def test(num_exp_id=0, 
         num_batch_size=64, 
         num_test_fold=0,
         model=resnet18,
         weights=None,
         num_class=3,
         output_dir="../result"
         ):
    print("Testing...")
    
    # setting hyper parameters
    BATCH_SIZE = num_batch_size
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512))
        ])

    # load data
    test_ds = baseline_dataset.BaselineDataset(
        "../data/csv/data_divide.csv", 
        transform=transform, 
        mode="test", 
        num_test_fold=num_test_fold,
        )
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # setting device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define models
    model = model(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.load_state_dict(torch.load(f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/weights/baseline.pt"))
    model.to(device)
    
    # define loss function
    criterion = nn.CrossEntropyLoss()
    
    # test
    model.eval()

    probabilities_list = []
    
    loss_list = []
    acc_list = []
    pred_list = []
    true_list = []
    basename_list = []
    
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data in test_dl:
            inputs, labels, basenames = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probabilities = torch.softmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append data to lists
            basename_list.extend(basenames)
            probabilities_list.extend(probabilities.tolist())
            pred_list.extend(predicted.tolist())
            true_list.extend(labels.tolist())
        
        test_loss = running_loss / len(test_dl)
        acc = correct / total

        loss_list.append(test_loss)
        acc_list.append(acc)
        
        print(f"loss: {test_loss:.4f}, accuracy: {acc:.4f}")
    
    malignant = np.array(probabilities_list)[:, 0]
    not_malignant = np.array(probabilities_list)[:, 1] + np.array(probabilities_list)[:, 2]
    pred_list = [1 if x == 2 else x for x in pred_list]
    true_list = [1 if x == 2 else x for x in true_list]
    
    data = {
        "basename": basename_list,
        "malignant": malignant,
        "not_malignant": not_malignant,
        "pred": pred_list,
        "true": true_list
        }
    
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/exp{num_exp_id}/fold_{num_test_fold}/csv/baseline_result.csv", index=False)
