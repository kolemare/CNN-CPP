import os
import json
import requests
import shutil
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

@task
def clean(ctx, build=False, datasets=False, all=False):
    """
    Clean the project.

    :param ctx: Context instance (automatically passed by Invoke).
    :param build: If True, clean the build directories.
    :param datasets: If True, clean the datasets directory.
    :param all: If True, clean both build and datasets subdirectories.
    """
    cmd = "./clean.sh"
    if all:
        cmd += " --all"
    else:
        if build:
            cmd += " --build"
        if datasets:
            cmd += " --datasets"
    ctx.run(cmd, pty=True)

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
