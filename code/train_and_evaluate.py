#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MobileShuffleNet
from data_preparation import train_loader, test_loader
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt

#replace it for different models
model = MobileShuffleNet(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, num_epochs=25):
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
    return losses

def plot_loss_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Loss Curve during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

loss_history = train_model(model, criterion, optimizer, train_loader, 25)
plot_loss_curve(loss_history)

def test_model():
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions, average='binary') * 100
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
    print(f'Recall on test set: {recall:.2f}%')
    print(f'F1 Score on test set: {f1:.2f}')
    
test_model()

