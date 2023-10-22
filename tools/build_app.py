import PyInstaller.__main__
import pip
import sys
import pkg_resources
import os
import shutil

ROOT_PATH = os.path.join("/", "tmp", "mnist-vae")
WORK_PATH = os.path.join(ROOT_PATH, "build")
DIST_PATH = os.path.join(ROOT_PATH, "dist")
SPEC_PATH = os.path.join(ROOT_PATH, "main.spec")

if os.path.exists(WORK_PATH):
    shutil.rmtree(WORK_PATH)
if os.path.exists(DIST_PATH):
    shutil.rmtree(DIST_PATH)
if os.path.exists(SPEC_PATH):
    os.remove(SPEC_PATH)


if sys.platform == "darwin":
    pip.main(["install", "--upgrade", "-r", "requirements.txt"])
elif sys.platform == "linux":
    pip.main(
        [
            "install",
            "--upgrade",
            "-r",
            "requirements.txt",
            "jax[cuda12_pip]",
            "-f",
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        ]
    )
else:
    raise NotImplementedError(
        "Windows is not supported because you can't easily install JAX"
    )


recursive_copy_metadata = ["neptune"]
collect_all_packages = []
if sys.platform == "linux":
    collect_all_packages.append("nvidia")

jax_location = None

for package in pkg_resources.working_set:
    if sys.platform == "linux":
        if any(maybe_needed in package.project_name for maybe_needed in ["nvidia"]):
            recursive_copy_metadata.append(package.project_name)

    if "jax" == package.project_name:
        jax_location = os.path.join(package.location, package.project_name)
        jax_version = package.version
        collect_all_packages.append(package.project_name)

if jax_location is None:
    raise RuntimeError(
        "JAX could not be found but is required to build the application, have you installed the requirements before running?"
    )
# else...


pyinstall_command = [
    "main.py",
    "--specpath",
    os.path.dirname(SPEC_PATH),
    "--name",
    "MNIST-VAE",
    "--onedir",
    "--windowed",
    "--noupx",
    "--distpath",
    DIST_PATH,
    "--workpath",
    WORK_PATH,
    "--add-data",
    f"{os.path.join(os.getcwd(), 'assets')}:assets",
    "--icon",
    os.path.join(os.getcwd(), "assets", "icon.png"),
]
for package_name in recursive_copy_metadata:
    pyinstall_command.append("--recursive-copy-metadata")
    pyinstall_command.append(package_name)

for package_name in collect_all_packages:
    pyinstall_command.append("--collect-all")
    pyinstall_command.append(package_name)


jax_mlir_filepath = os.path.join(jax_location, "_src", "interpreters", "mlir.py")
if not os.path.exists(jax_mlir_filepath):
    raise RuntimeError(
        f"Could not find {jax_mlir_filepath} which is expected in order to apply a patch!"
    )
# else...
with open(jax_mlir_filepath, "r") as f:
    mlir_lines = f.readlines()

new_mlir_lines = mlir_lines.copy()
# Commenting these lines worked fine for https://github.com/google/jax/issues/17705,
#  in order to get JAX to run when inside app.
if jax_version == "0.4.19":
    comment_out_indices = range(822, 839)
elif jax_version == "0.4.11":
    comment_out_indices = range(650, 659)
elif jax_version == "0.4.13":
    comment_out_indices = range(709, 718)
else:
    raise NotImplementedError(
        f"Patching for JAX version {jax_version} is not supported!"
    )
for line_index in comment_out_indices:
    new_mlir_lines[line_index] = "#" + new_mlir_lines[line_index]


try:
    with open(jax_mlir_filepath, "w") as f:
        f.writelines(new_mlir_lines)

    PyInstaller.__main__.run(pyinstall_command)
finally:
    with open(jax_mlir_filepath, "w") as f:
        f.writelines(mlir_lines)
