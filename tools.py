import os
import shutil
import csv
import subprocess
import zipfile
import matplotlib.pyplot as plt

def clean_build():
    print("Cleaning build directory...")
    if os.path.exists('build'):
        shutil.rmtree('build')

def delete_txts():
    print("Cleaning all .txt files...")
    for item in os.listdir('.'):
        if item.endswith('.txt') and item != 'CMakeLists.txt':
            os.remove(item)

def delete_pngs_and_csvs():
    print("Cleaning all .png and .csv files in logs and plots directories...")
    for directory in ['logs', 'plots']:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                if item.endswith('.png') or item.endswith('.csv'):
                    os.remove(os.path.join(directory, item))

def delete_vscode_folder():
    vscode_folder = '.vscode'
    if os.path.exists(vscode_folder):
        shutil.rmtree(vscode_folder)

def clean_datasets():
    """
    Clean the datasets folder.
    """
    print("Cleaning datasets directory...")
    datasets_dir = 'datasets'
    if os.path.exists(datasets_dir):
        for item in os.listdir(datasets_dir):
            item_path = os.path.join(datasets_dir, item)
            if not (item.endswith('.zip') or '.zip.' in item):
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

def extract_datasets(directory):
    """
    Extract all .zip files in the datasets folder.
    It handles both regular zip files and split zip files, and deletes the combined zip files after extraction.
    
    :param directory: The directory where the zip files are located.
    """
    if not os.path.exists(directory):
        print("Datasets directory does not exist. Skipping extraction.")
        return

    # Dictionary to store split parts of zip files
    split_files = {}

    # Identify regular and split zip files
    for item in os.listdir(directory):
        if item.endswith('.zip'):
            # Handle regular zip files
            zip_path = os.path.join(directory, item)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            print(f"Extracted {zip_path} in {directory}.")
        elif '.zip.' in item:
            # Identify split files
            base_name = item.split('.zip.')[0] + '.zip'
            if base_name not in split_files:
                split_files[base_name] = []
            split_files[base_name].append(item)

    # Combine and extract split zip files
    for base_name, parts in split_files.items():
        parts = sorted(parts)
        combined_zip_path = os.path.join(directory, base_name)
        with open(combined_zip_path, 'wb') as combined_file:
            for part in parts:
                part_path = os.path.join(directory, part)
                with open(part_path, 'rb') as part_file:
                    shutil.copyfileobj(part_file, combined_file)
        print(f"Combined split files into {combined_zip_path}")
        
        # Extract the combined zip file
        if zipfile.is_zipfile(combined_zip_path):
            with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            print(f"Extracted {combined_zip_path} in {directory}.")
            
            # Delete the combined zip file after extraction
            os.remove(combined_zip_path)
            print(f"Deleted combined zip file: {combined_zip_path}")
        else:
            print(f"Error: {combined_zip_path} is not a valid zip file.")

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

# Function to determine color based on elrales state
def get_color(elrales):
    if elrales == 'NORMAL' or elrales == 'OFF':
        return (82/255, 127/255, 199/255)  # Normal mode
    elif elrales == 'RECOVERY':
        return (0, 128/255, 0)  # Recovery mode (green)
    elif elrales == 'LOSING':
        return (1, 0, 0)  # Losing mode (red)
    elif elrales == 'EARLY_STOPPING':
        return (0, 0, 0)  # Early stopping mode (black)
    return (82/255, 127/255, 199/255)  # Default to normal mode color

# Function to generate and save the plots
def generate_plots(epoch_data, output_dir, base_filename):
    epochs = [data['epoch'] for data in epoch_data]
    training_accuracy = [data['training_accuracy'] for data in epoch_data]
    training_loss = [data['training_loss'] for data in epoch_data]
    testing_accuracy = [data['testing_accuracy'] for data in epoch_data]
    testing_loss = [data['testing_loss'] for data in epoch_data]

    # Plot 1: CNN Accuracy
    plt.figure()
    for i in range(len(epochs) - 1):
        color = get_color(epoch_data[i]['elrales'])
        plt.plot(epochs[i:i+2], training_accuracy[i:i+2], color=color)
    plt.plot(epochs, testing_accuracy, label='Testing Accuracy', color=(148/255, 0/255, 211/255))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Accuracy')
    plt.legend()
    plt.savefig(f"{output_dir}/cnn_accuracy.png")

    # Plot 2: CNN Loss
    plt.figure()
    for i in range(len(epochs) - 1):
        color = get_color(epoch_data[i]['elrales'])
        plt.plot(epochs[i:i+2], training_loss[i:i+2], color=color)
    plt.plot(epochs, testing_loss, label='Testing Loss', color=(148/255, 0/255, 211/255))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/cnn_loss.png")

def build_project(jobs):
    print("Building CNN-CPP...")
    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    cmake_command = f'cmake -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF ..'
    subprocess.run(cmake_command, shell=True, check=True)
    make_command = f'make -j{jobs}'
    subprocess.run(make_command, shell=True, check=True)
    os.chdir('..')
    print("Build completed.")

def run_tests():
    print("Running tests...")
    if not os.path.exists('build'):
        print("Build directory does not exist. Please build the project first.")
        return
    os.chdir('build')
    if os.path.isfile('./CNN_CPP'):
        test_command = './CNN_CPP --tests'
        subprocess.run(test_command, shell=True, check=True)
    else:
        print("Test executable not found. Please build the project first.")
    os.chdir('..')
    print("Tests run completed.")
