#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import paramiko
import os
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import torchvision.transforms as transforms

# Function to perform the classification
def classify_images():
    start_time = time.time()
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Variables to store images
    all_images = []
    artificial_images = []
    
    # Classify images and separate those predicted as "Artificial"
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(images)):
                img = images[i].cpu()
                all_images.append(img)
                if predicted[i].item() == 0:  # Assuming 0 is the label for "Artificial"
                    artificial_images.append(img)

    end_time = time.time()
    classification_time = end_time - start_time
    return classification_time, len(all_images), len(artificial_images)

# List to store times for each classification
times = []

# Perform classification 10 times and record times
for _ in range(10):
    time_taken, total_images, artificial_count = classify_images()
    times.append(time_taken)
    print(f'Classification time: {time_taken:.4f} seconds')

# Calculate and print the average time
average_time = sum(times) / len(times)
print(f'Average classification time: {average_time:.4f} seconds')
print(f'Total images per run: {total_images}')
print(f'Artificial images per run: {artificial_count}')

# Variables to store images
all_images = []
artificial_images = []
    
# Classify images and separate those predicted as "Artificial"
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(images)):
            img = images[i].cpu()
            all_images.append(img)
            if predicted[i].item() == 0:  # Assuming 0 is the label for "Artificial"
                artificial_images.append(img) 
                


# file create
if not os.path.exists('all_images'):
    os.makedirs('all_images')
if not os.path.exists('artificial_images'):
    os.makedirs('artificial_images')

# save as  all_images
for idx, img in enumerate(all_images):
    img = transforms.ToPILImage()(img)
    img.save(f'all_images/img_{idx}.jpeg')

# save as artificial_images
for idx, img in enumerate(artificial_images):
    img = transforms.ToPILImage()(img)
    img.save(f'artificial_images/img_{idx}.jpeg')


def transfer_files(local_dir, remote_dir, hostname, port, username, password, delay=0.1):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(hostname, port, username, password)
        sftp = ssh_client.open_sftp()

        # Ensure remote directory exists or create it
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            sftp.mkdir(remote_dir)

        total_start_time = time.time()
        total_bytes = 0
        file_sizes = []
        transfer_times = []
        timestamps = []

        # Iterate through local directory and upload each file
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            remote_path = os.path.join(remote_dir, filename.replace("\\", "/"))
            if os.path.isfile(local_path):
                file_info = os.stat(local_path)
                total_bytes += file_info.st_size
                start_time = time.time()
                sftp.put(local_path, remote_path)  # Upload file
                end_time = time.time()
                file_sizes.append(file_info.st_size)
                transfer_times.append(end_time - start_time)
                timestamps.append(end_time - total_start_time)
                time.sleep(delay)  # Introduce delay to simulate slower network

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        transfer_speed = total_bytes / total_time / 1024 / 1024

        print(f'Total transfer time for {local_dir} to {remote_dir}: {total_time:.2f} seconds')
        print(f'Total data transferred: {total_bytes / 1024 / 1024:.2f} MB')
        print(f'Average transfer speed: {transfer_speed:.2f} MB/s')

        return file_sizes, transfer_times, timestamps
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        sftp.close()
        ssh_client.close()

# Function to perform multiple transfers and calculate average
# modify"your route" "your hostname" "your username"
def perform_multiple_transfers(n=10):
    cumulative_session_time = 0
    for _ in range(n):
        start_time = time.time()
        transfer_files(
            'artificial_images', 'your route', 'your hostname', 22, 'your username', '  ', delay=0.001)
        session_time = time.time() - start_time
        cumulative_session_time += session_time
        print(f'Total transfer time for iteration: {session_time:.4f} seconds')
    average_session_time = cumulative_session_time / n
    print(f'Average total transfer time over {n} runs: {average_session_time:.4f} seconds')

# This function will now correctly calculate the average of the total transfer times.
perform_multiple_transfers()

