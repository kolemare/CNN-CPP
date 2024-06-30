from invoke import task

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
    cmd = "./build.sh"
    if clean:
        cmd += " --clean"
    cmd += f" -j{jobs}"
    ctx.run(cmd, pty=True)


@task
def install(ctx):
    ctx.run("./install.sh", pty=True)


@task
def test(ctx):
    ctx.run("./run_tests.sh", pty=True)


@task
def run(ctx):
    ctx.run("./build/CNN_CPP", pty=True)


@task
def download(ctx, clean=False):
    cmd = "./download.sh"
    if clean:
        cmd += " --clean"
    ctx.run(cmd, pty=True)
