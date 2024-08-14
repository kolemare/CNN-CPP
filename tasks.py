import os
import subprocess
from invoke import task
from tools import (
    clean_build,
    delete_txts,
    delete_pngs_and_csvs,
    parse_csv_file,
    generate_plots,
    build_project,
    run_tests,
    delete_vscode_folder,
    clean_datasets,
    extract_datasets
)

@task
def clean(ctx, build=False, datasets=False):
    """
    Clean the project.

    :param ctx: Context instance (automatically passed by Invoke).
    :param build: If True, clean the build directories.
    :param datasets: If True, delete everything in the datasets folder except .zip files.
    """
    if build:
        clean_build()

    if datasets:
        clean_datasets()

    delete_txts()
    delete_pngs_and_csvs()
    delete_vscode_folder()

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
def plot(ctx, csv="logs/cnn.csv", output_dir='plots'):
    """
    Parse the CSV file and generate plots for training and testing metrics.

    :param ctx: Context instance (automatically passed by Invoke).
    :param csv: Path to the CSV file.
    :param output_dir: Directory where plots will be saved.
    """
    epoch_data = parse_csv_file(csv)
    if epoch_data:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(csv))[0]
        generate_plots(epoch_data, output_dir, base_filename)
        print(f"Plots saved to {output_dir}")
    else:
        print("No epoch data to plot.")

@task
def extract(ctx):
    """
    Extract all .zip files in the datasets folder.

    :param ctx: Context instance (automatically passed by Invoke).
    """
    extract_datasets('datasets')
