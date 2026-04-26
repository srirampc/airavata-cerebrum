# Apache Airavata Cerebrum, an Integrated Neuroscience Computational Framework

## Introduction

Airavata Cerebrum aims to simplify building whole brain models via integration
of cellular-level brain atlases with advanced computational tools.
Our goal is to create a cohesive, open-source framework that allows for
accelerated 'data to model' workflows that are flexible to update and
straight-forward to reproduce.

## Key Features

- **Integration of Brain Atlases**: Transparently connect to publicly available
  cellular-level brain atlases into a single, accessible platform.
- **Model Consruction Workflows**: Workflows to collect/filter/combine data from
  different databases and construct whole brain scale models.
- **Open-Source Framework**: Developing a user-friendly, open-source environment
  for neuroscience research.
- **Streamlined Environment**: Ensuring a lightweight, efficient framework for
  both beginners and advanced users alike.

# Install Airavata Cerebrum

Airavata Cerebrum requires python3.10+ environment.
It is currently tested only in Linux operating system.
To install from the source, we recommend creating a `conda` environment using
[miniforge](https://github.com/conda-forge/miniforge) as below:

```
conda config --add channels conda-forge
conda create -n cerebrum python=3.10 nodejs
conda activate cerebrum
```

To install Airavata Cerebrum from source into the environment created above,
pip can be used with the git option as follows.

```
pip install git+https://github.com/apache/airavata-cerebrum.git
```

# Model Notebooks and Scripts

The `resources` directory contains a set of notebooks to demonstrate Cerebrum, and
also standalone batch scripts that build/simulate models using cerbrum.

## IPython Notebooks

| Model                                     | Notebook                                                          |
| ----------------------------------------- | ----------------------------------------------------------------- |
| Demo of Cerebrum V1L4 model               | [V1 L4 IPython Notebook](resources/notebooks/V1L4-Notebook.ipynb) |
| Demo of Cerebrum V1 model                 | [V1 IPython Notebook](resources/notebooks/V1-Notebook.ipynb)      |
| Demo of Cerebrum V1 model on Cybershuttle | [V1 IPython Notebook](resources/notebooks/V1-CS-Notebook.ipynb)   |
| Demo of WGN Sleep model                   | [WGN Sleep IPython Notebook](resources/notebooks/WGN-Sleep.ipynb) |

## Command-line scripts

| Model                              | Scripts                                                           |
| ---------------------------------- | ----------------------------------------------------------------- |
| Build/Simulate Cerebrum V1 model   | [V1 script](resources/notebooks/mousev1/cli.py)                   |
| Simulate Cerebrum V1 model w. BMTK | [V1 script](resources/notebooks/mousev1/simulate_cli.py)          |
| Simulate Cerebrum V1 model w. NEST | [V1 script](resources/notebooks/mousev1/nest_simulate_cli.py)     |


Please refer to [resources/README.md](resources/notebooks/README.md) for additional 
installation requirements to run the notebooks. 


# Development 

## Installing Environment For Development

Development environment for Airavata Cerebrum can be created as a python3.10+ 
virtual environment in conda using the `environment.yml` file.
```
conda create env -n cerebrum -f environment.yml
conda activate cerebrum
```

`environment.yml` includes the version of each of the package to make conda's 
dependency resolution algorithm to run faster. 

## Potential Environment Issues

### Miniforge `conda` Issue when installed with `spack`

If installing conda with miniforge, in some cases the following error appears:
`ModuleNotFoundError: No module named 'conda'`

This happens when the conda script (`$CONDA_EXE`)  has in its first line 
`#!/usr/bin/env python`, which picks up the python from the `cerebrum` environment
which is currently being installed instead of the base environment. To fix this,
replace the `/usr/bin/env python` with the python installed in the base 
environment (generally corresponds of the environment variable `$CONDA_PYTHON_EXE`).
