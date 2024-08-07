import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from data_preparation import test_loader, artificial_classes, natural_classes

def imshow(img, title):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def show_sample_images(loader, class_names):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    for i in range(len(labels)):
        label = labels[i].item()
        category_label = "artificial" if class_names[label] in artificial_classes else "natural"
        imshow(torchvision.utils.make_grid(images[i]), category_label + " - ")

def imshow_and_save(images, titles, filename):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 2x2 grid
    axs = axs.flatten()

    for i, (img, ax) in enumerate(zip(images, axs)):
        # Normalize the tensor image data to [0, 1]
        img = img / 2 + 0.5  # assuming the image was normalized with mean=0.5 and std=0.5
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))  # Convert to HxWxC format for matplotlib
        ax.imshow(npimg)
        ax.set_title(titles[i], fontsize=12)  # Adjust font size as needed
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

# Set the model to evaluation mode
model.eval()

# Get a batch of images from the test set
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Predict
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Shuffle indices
shuffled_indices = torch.randperm(len(images))

# Process each image individually to create a list of individual images
# Select the first four shuffled indices for display
images_processed = [torchvision.utils.make_grid(images[idx].unsqueeze(0)) for idx in shuffled_indices[:4]]
labels_titles = [f'True: {"Artificial" if labels[idx].item() == 0 else "Natural"} / Pred: {"Artificial" if predicted[idx].item() == 0 else "Natural"}' for idx in shuffled_indices[:4]]

# Display the images with embedded titles and save them
imshow_and_save(images_processed, labels_titles, 'concatenated_images_with_labels.png')
