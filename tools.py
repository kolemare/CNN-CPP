import os
import shutil
import csv
import subprocess
import zipfile
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple

def clean_build():
    print("Cleaning build directory...")
    if os.path.exists('build'):
        shutil.rmtree('build')

def delete_txts():
    print("Cleaning all .txt files...")
    for item in os.listdir('.'):
        if item.endswith('.txt') and item != 'CMakeLists.txt' and item != 'LICENSE.txt':
            os.remove(item)

def delete_pngs_and_csvs():
    print("Cleaning all .png and .csv files in logs and plots directories...")
    for directory in ['logs', 'plots']:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                if item.endswith('.png') or item.endswith('.csv') or item.endswith('.txt'):
                    os.remove(os.path.join(directory, item))

def delete_docs_vscode():
    print("Cleaning docs and IDE specifics...")
    for directory in ['.vscode', 'docs']:
        if os.path.exists(directory):
            shutil.rmtree(directory)

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
                'validation_accuracy': float(row['validation_accuracy']),
                'validation_loss': float(row['validation_loss']),
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
    validation_accuracy = [data['validation_accuracy'] for data in epoch_data]
    validation_loss = [data['validation_loss'] for data in epoch_data]

    # Determine if ELRALES is used
    elrales_states = {data['elrales'] for data in epoch_data}
    elrales_enabled = len(elrales_states - {'NORMAL', 'OFF'}) > 0

    # Plot 1: CNN Accuracy
    plt.figure()
    for i in range(len(epochs) - 1):
        color = get_color(epoch_data[i]['elrales'])
        plt.plot(epochs[i:i+2], training_accuracy[i:i+2], color=color)
    
    if epoch_data[-1]['elrales'] == 'EARLY_STOPPING':
        plt.plot(epochs[-1:], [training_accuracy[-1]], 'o', color='black')

    # Add agenda
    if elrales_enabled:
        plt.plot([], [], label='Normal', color=get_color('NORMAL'))
        plt.plot([], [], label='Recovery', color=get_color('RECOVERY'))
        plt.plot([], [], label='Losing', color=get_color('LOSING'))
        plt.scatter([], [], label='Early Stopping', color='black')  # Black ball in the legend
    else:
        plt.plot([], [], label='Training Accuracy', color=(82/255, 127/255, 199/255))

    plt.plot(epochs, validation_accuracy, label='Validation Accuracy', color=(148/255, 0/255, 211/255))
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_accuracy.png")

    # Plot 2: CNN Loss
    plt.figure()
    for i in range(len(epochs) - 1):
        color = get_color(epoch_data[i]['elrales'])
        plt.plot(epochs[i:i+2], training_loss[i:i+2], color=color)
    
    if epoch_data[-1]['elrales'] == 'EARLY_STOPPING':
        plt.plot(epochs[-1:], [training_loss[-1]], 'o', color='black')  # Plot a black ball for Early Stopping

    # Add agenda
    if elrales_enabled:
        plt.plot([], [], label='Normal', color=get_color('NORMAL'))
        plt.plot([], [], label='Recovery', color=get_color('RECOVERY'))
        plt.plot([], [], label='Losing', color=get_color('LOSING'))
        plt.scatter([], [], label='Early Stopping', color='black')  # Black ball in the legend
    else:
        plt.plot([], [], label='Training Loss', color=(82/255, 127/255, 199/255))

    plt.plot(epochs, validation_loss, label='Validation Loss', color=(148/255, 0/255, 211/255))
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/{base_filename}_loss.png")

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

def generate_pdf():
    # Run Doxygen to generate LaTeX files
    print("Running Doxygen...")
    subprocess.run("doxygen Doxyfile", shell=True, check=True)

    # Navigate to the LaTeX output directory
    latex_dir = "docs/latex"
    if not os.path.exists(latex_dir):
        print("LaTeX directory does not exist. Make sure Doxygen generated the LaTeX files.")
        return

    # Compile the LaTeX files into a PDF
    print("Compiling LaTeX files to PDF...")
    subprocess.run(f"cd {latex_dir} && make pdf", shell=True, check=True)

    # Inform the user where the PDF is located
    pdf_path = os.path.join(latex_dir, "refman.pdf")
    if os.path.exists(pdf_path):
        print(f"PDF generated successfully and located at: {pdf_path}")
    else:
        print("Failed to generate the PDF. Please check for LaTeX errors.")


def plot_eval_json(json_path: str, out_dir: str, top_k: int = 15) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r") as f:
        metrics: Dict[str, float] = json.load(f)

    class_names = _find_classes(metrics)
    if not class_names:
        raise RuntimeError("plot_eval_json: couldn't find class names (expected keys like 'class/<name>/recall').")

    class_names = sorted(class_names)
    idx = {name: i for i, name in enumerate(class_names)}
    C = len(class_names)

    confusion = [[0.0] * C for _ in range(C)]
    for key, val in metrics.items():
        if not isinstance(key, str) or not key.startswith("confusion/true="):
            continue

        tname, pname = _split_conf_key(key)
        if tname is None or pname is None:
            continue
        if tname not in idx or pname not in idx:
            continue

        confusion[idx[tname]][idx[pname]] = float(val)

    if sum(sum(r) for r in confusion) <= 0:
        raise RuntimeError("plot_eval_json: confusion matrix not found or empty.")

    # 1) confusion heatmap (row-normalized)
    row_norm = _normalize_rows(confusion)
    _save_confusion_heatmap(
        row_norm,
        class_names,
        os.path.join(out_dir, "confusion_row_normalized.png"),
    )

    # 2) per-class recall
    recalls = [float(metrics.get(f"class/{name}/recall", 0.0)) for name in class_names]
    _save_bar_chart(
        values=recalls,
        labels=class_names,
        title="Per-class Recall",
        ylabel="Recall",
        out_path=os.path.join(out_dir, "per_class_recall.png"),
        ylim=(0.0, 1.0),
    )

    # 3) top off-diagonal confusions
    pairs = []
    for i, tname in enumerate(class_names):
        for j, pname in enumerate(class_names):
            if i == j:
                continue
            v = confusion[i][j]
            if v > 0:
                pairs.append((v, tname, pname))

    pairs.sort(key=lambda x: x[0], reverse=True)
    pairs = pairs[:max(1, top_k)]

    if pairs:
        vals = [p[0] for p in pairs]
        labels = [f"{p[1]} → {p[2]}" for p in pairs]
        _save_barh_chart(
            values=vals,
            labels=labels,
            title=f"Top {len(pairs)} Confusions",
            xlabel="Count",
            out_path=os.path.join(out_dir, "top_confusions.png"),
        )
    else:
        # If there are no off-diagonal entries (perfect model), still emit an image
        _save_barh_chart(
            values=[0],
            labels=["(none)"],
            title="Top Confusions",
            xlabel="Count",
            out_path=os.path.join(out_dir, "top_confusions.png"),
        )


def _find_classes(metrics: Dict[str, float]) -> List[str]:
    names = set()
    for k in metrics.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("class/") and k.endswith("/recall"):
            parts = k.split("/")
            if len(parts) == 3:
                names.add(parts[1])
    return list(names)


def _split_conf_key(key: str) -> Tuple[str | None, str | None]:
    # "confusion/true=airplane/pred=ship"
    parts = key.split("/")
    if len(parts) != 3:
        return None, None

    tpart, ppart = parts[1], parts[2]
    if not tpart.startswith("true=") or not ppart.startswith("pred="):
        return None, None

    return tpart[5:], ppart[5:]


def _normalize_rows(mat: List[List[float]]) -> List[List[float]]:
    out = []
    for row in mat:
        s = float(sum(row))
        if s <= 0.0:
            out.append([0.0] * len(row))
            continue
        out.append([v / s for v in row])
    return out


def _save_confusion_heatmap(mat: List[List[float]], labels: List[str], out_path: str) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    ticks = range(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_bar_chart(
    values: List[float],
    labels: List[str],
    title: str,
    ylabel: str,
    out_path: str,
    ylim: Tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x, labels, rotation=45, ha="right")
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_barh_chart(values: List[float], labels: List[str], title: str, xlabel: str, out_path: str) -> None:
    plt.figure(figsize=(12, 7))
    y = range(len(labels))
    plt.barh(y, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()