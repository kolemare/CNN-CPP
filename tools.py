import os
import requests
import shutil
import csv
import matplotlib.pyplot as plt

# Function to download and unzip a dataset
def download_and_unzip(url, target_dir):
    local_filename = os.path.join(target_dir, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    # Unzip the file
    os.system(f'unzip {local_filename} -d {target_dir}')
    # Remove the zip file
    os.remove(local_filename)

def clean_build():
    print("Cleaning build directory and submodules...")
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('external/opencv/build'):
        shutil.rmtree('external/opencv/build')
    if os.path.exists('external/googletest/build'):
        shutil.rmtree('external/googletest/build')
    os.system('git submodule foreach --recursive git clean -fdx')
    os.system('git submodule foreach --recursive git reset --hard')
    print("Build clean completed.")

def clean_datasets():
    print("Cleaning datasets directory...")
    if os.path.exists('datasets'):
        for item in os.listdir('datasets'):
            if item != '.gitkeep':
                item_path = os.path.join('datasets', item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        print("Datasets clean completed.")
    else:
        print("Datasets directory does not exist. Skipping.")

def delete_txts():
    print("Deleting all .txt files...")
    for item in os.listdir('.'):
        if item.endswith('.txt') and item != 'CMakeLists.txt':
            os.remove(item)
    print("TXT files deletion completed.")

def delete_pngs_and_csvs():
    print("Deleting all .png and .csv files in logs and plots directories...")
    for directory in ['logs', 'plots']:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                if item.endswith('.png') or item.endswith('.csv'):
                    os.remove(os.path.join(directory, item))
    print("PNG and CSV files deletion completed.")

# Function to parse the CSV file and extract relevant data
def parse_csv_file(csv_path):
    epoch_data = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            epoch_data.append({
                'epoch': int(row['epoch_num']),
                'training_accuracy': float(row['training_accuracy']),
                'training_loss': float(row['training_loss']),
                'testing_accuracy': float(row['testing_accuracy']),
                'testing_loss': float(row['testing_loss']),
                'elrales': row['elrales'].strip(),
            })
    
    return epoch_data

# Function to save the parsed data to a CSV file
def save_to_csv(epoch_data, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'training_accuracy', 'training_loss', 'testing_accuracy', 'testing_loss', 'elrales']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in epoch_data:
            writer.writerow(data)

# Function to generate and save the plots
def generate_plots(epoch_data, output_dir, base_filename, mode='elrales'):
    if mode == 'normal':
        # Filter data to include only NORMAL or OFF epochs
        filtered_data = [data for data in epoch_data if data['elrales'] in ['NORMAL', 'OFF']]
        # Reset epochs to be consecutive
        for i, data in enumerate(filtered_data):
            data['epoch'] = i + 1
    else:
        # Use the data as is for elrales mode
        filtered_data = epoch_data

    epochs = [data['epoch'] for data in filtered_data]
    training_accuracy = [data['training_accuracy'] for data in filtered_data]
    training_loss = [data['training_loss'] for data in filtered_data]
    testing_accuracy = [data['testing_accuracy'] for data in filtered_data]
    testing_loss = [data['testing_loss'] for data in filtered_data]

    # Plot training loss and accuracy
    plt.figure()
    plt.plot(epochs, training_loss, label='Training Loss', color='orange')
    plt.plot(epochs, training_accuracy, label='Training Accuracy', color='lightblue')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(f'Training Loss and Accuracy over Epochs ({mode})')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_training_metrics_{mode}.png")

    # Plot testing loss and accuracy
    plt.figure()
    plt.plot(epochs, testing_loss, label='Testing Loss', color='lightblue')
    plt.plot(epochs, testing_accuracy, label='Testing Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title(f'Testing Loss and Accuracy over Epochs ({mode})')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_testing_metrics_{mode}.png")

    # Plot combined training and testing accuracy
    plt.figure()
    plt.plot(epochs, training_accuracy, label='Training Accuracy', color='lightblue', linestyle='-')
    plt.plot(epochs, testing_accuracy, label='Testing Accuracy', color='orange', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Testing Accuracy over Epochs ({mode})')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_combined_accuracy_{mode}.png")

    # Plot combined training and testing loss
    plt.figure()
    plt.plot(epochs, training_loss, label='Training Loss', color='orange', linestyle='-')
    plt.plot(epochs, testing_loss, label='Testing Loss', color='lightblue', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Testing Loss over Epochs ({mode})')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_combined_loss_{mode}.png")
