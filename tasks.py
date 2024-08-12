import os
import json
import shutil
import csv
import requests
import subprocess
from invoke import task
import matplotlib.pyplot as plt
from tools import (
    clean_build,
    clean_datasets,
    delete_txts,
    delete_pngs_and_csvs,
    parse_csv_file,
    save_to_csv,
    generate_plots,
    download_and_unzip,
    build_project,
    run_tests,
    delete_vscode_folder
)

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
    delete_vscode_folder()  # Delete the .vscode folder

@task
def build(ctx, clean=False, jobs=1):
    """
    Build the project.

    :param ctx: Context instance (automatically passed by Invoke).
    :param clean: If True, clean the build directories before building.
    :param jobs: Number of jobs to run simultaneously (default is 1).
    """
    if clean:
        clean_build()
    build_project(jobs)

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
    run_tests()

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

@task
def extract_to_csv(ctx, log_file, output_dir):
    """
    Parse the log file and save the relevant data to a CSV file.

    :param ctx: Context instance (automatically passed by Invoke).
    :param log_file: Path to the log file.
    :param output_dir: Directory where the CSV file will be saved.
    """
    epoch_data = parse_csv_file(log_file)
    if epoch_data:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(log_file))[0]
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        save_to_csv(epoch_data, csv_path)
        print(f"Data saved to {csv_path}")
    else:
        print("No epoch data to save.")

@task
def plot_from_csv(ctx, csv_file, output_dir, mode='normal'):
    """
    Parse the CSV file and generate plots for training and testing metrics.

    :param ctx: Context instance (automatically passed by Invoke).
    :param csv_file: Path to the CSV file.
    :param output_dir: Directory where plots will be saved.
    :param mode: Plotting mode, either 'normal' or 'elrales'.
    """
    epoch_data = parse_csv_file(csv_file)
    if epoch_data:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(csv_file))[0]
        generate_plots(epoch_data, output_dir, base_filename, mode)
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
    delete_vscode_folder()
