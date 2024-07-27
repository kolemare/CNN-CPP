import os
import json
import requests
import shutil
import re
import csv
import matplotlib.pyplot as plt
from invoke import task

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

@task
def clean(ctx, build=False, datasets=False, all=False):
    """
    Clean the project.

    :param ctx: Context instance (automatically passed by Invoke).
    :param build: If True, clean the build directories.
    :param datasets: If True, clean the datasets directory.
    :param all: If True, clean both build and datasets subdirectories.
    """
    if all:
        clean_build()
        clean_datasets()
    else:
        if build:
            clean_build()
        if datasets:
            clean_datasets()
    delete_txts()
    delete_pngs_and_csvs()

@task
def build(ctx, clean=False, jobs=1):
    """
    Build the project.

    :param ctx: Context instance (automatically passed by Invoke).
    :param clean: If True, clean the build directories before building.
    :param jobs: Number of jobs to run simultaneously (default is 1).
    """
    cmd = "./build.sh"
    if clean:
        cmd += " --clean"
    cmd += f" -j{jobs}"
    ctx.run(cmd, pty=True)

@task
def install(ctx):
    """
    Install the project.

    :param ctx: Context instance (automatically passed by Invoke).
    """
    ctx.run("./install.sh", pty=True)

@task
def test(ctx):
    """
    Run tests for the project.

    :param ctx: Context instance (automatically passed by Invoke).
    """
    ctx.run("./run_tests.sh", pty=True)

@task
def run(ctx):
    """
    Run the project.

    :param ctx: Context instance (automatically passed by Invoke).
    """
    ctx.run("./build/CNN_CPP", pty=True)

@task
def download(ctx, clean=False, config='datasets.json'):
    """
    Download datasets as specified in the config file.

    :param ctx: Context instance (automatically passed by Invoke).
    :param clean: If True, clean the datasets directory before downloading.
    :param config: Path to the JSON config file with dataset URLs.
    """
    target_dir = 'datasets'
    os.makedirs(target_dir, exist_ok=True)

    if clean:
        print("Cleaning up the datasets directory...")
        for item in os.listdir(target_dir):
            if item != '.gitkeep':
                item_path = os.path.join(target_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        print("Datasets directory cleaned.")

    # Read the JSON config file
    with open(config, 'r') as f:
        data = json.load(f)

    # Download each dataset specified in the config file
    for dataset in data['datasets']:
        print(f"Downloading {dataset['name']}...")
        download_and_unzip(dataset['url'], target_dir)
        print(f"{dataset['name']} downloaded and extracted to {target_dir}")

    print("All datasets downloaded successfully.")

# Function to parse the log file and extract relevant data
def parse_log_file(log_file):
    epoch_data = []
    current_epoch = None
    current_data = {}

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            epoch_match = re.match(r"Epoch (\d+) complete.", line)
            if epoch_match:
                if current_epoch is not None and current_data:
                    epoch_data.append(current_data)
                current_epoch = int(epoch_match.group(1))
                current_data = {'epoch': current_epoch}
                continue

            training_acc_match = re.match(r"Training Accuracy: ([\d\.]+)", line)
            if training_acc_match:
                current_data['training_accuracy'] = float(training_acc_match.group(1))
                continue

            avg_loss_match = re.match(r"Average Loss: ([\d\.]+)", line)
            if avg_loss_match:
                current_data['average_loss'] = float(avg_loss_match.group(1))
                continue

            testing_acc_match = re.match(r"Testing Accuracy: ([\d\.]+)", line)
            if testing_acc_match:
                current_data['testing_accuracy'] = float(testing_acc_match.group(1))
                continue

            testing_loss_match = re.match(r"Testing Loss\(avg\): ([\d\.]+)", line)
            if testing_loss_match:
                current_data['testing_loss'] = float(testing_loss_match.group(1))
                continue

    if current_epoch is not None and current_data:
        epoch_data.append(current_data)
    
    if not epoch_data:
        print("No data found in log file. Please check the format.")
    
    return epoch_data

# Function to save the parsed data to a CSV file
def save_to_csv(epoch_data, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'training_accuracy', 'average_loss', 'testing_accuracy', 'testing_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in epoch_data:
            writer.writerow(data)

# Function to generate and save the plots
def generate_plots(epoch_data, output_dir, base_filename):
    epochs = [data['epoch'] for data in epoch_data]
    training_accuracy = [data.get('training_accuracy') for data in epoch_data]
    training_loss = [data.get('average_loss') for data in epoch_data]
    testing_accuracy = [data.get('testing_accuracy') for data in epoch_data]
    testing_loss = [data.get('testing_loss') for data in epoch_data]

    # Plot training loss
    plt.figure()
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_training_loss.png")

    # Plot training accuracy
    plt.figure()
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_training_accuracy.png")

    # Plot testing loss
    plt.figure()
    plt.plot(epochs, testing_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Testing Loss over Epochs')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_testing_loss.png")

    # Plot testing accuracy
    plt.figure()
    plt.plot(epochs, testing_accuracy, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Testing Accuracy over Epochs')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_testing_accuracy.png")

@task
def extract_to_csv(ctx, log_file, output_dir):
    """
    Parse the log file and save the relevant data to a CSV file.

    :param ctx: Context instance (automatically passed by Invoke).
    :param log_file: Path to the log file.
    :param output_dir: Directory where the CSV file will be saved.
    """
    epoch_data = parse_log_file(log_file)
    if epoch_data:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(log_file))[0]
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        save_to_csv(epoch_data, csv_path)
        print(f"Data saved to {csv_path}")
    else:
        print("No epoch data to save.")

@task
def plot_from_log(ctx, log_file, output_dir):
    """
    Parse the log file and generate plots for training and testing metrics.

    :param ctx: Context instance (automatically passed by Invoke).
    :param log_file: Path to the log file.
    :param output_dir: Directory where plots will be saved.
    """
    epoch_data = parse_log_file(log_file)
    if epoch_data:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(log_file))[0]
        generate_plots(epoch_data, output_dir, base_filename)
        print(f"Plots saved to {output_dir}")
    else:
        print("No epoch data to plot.")

@task(default=True)
def default_clean(ctx):
    """
    Default clean action that deletes txt, png, and csv files.
    """
    delete_txts()
    delete_pngs_and_csvs()
