from invoke import task


@task
def build(ctx, clean=False, jobs=1):
    cmd = "./build.sh"
    if clean:
        cmd += " --clean"
    cmd += f" -j{jobs}"
    ctx.run(cmd, pty=True)


@task
def clean(ctx):
    ctx.run("./clean.sh", pty=True)


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
