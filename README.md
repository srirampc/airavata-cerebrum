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

# Model Notebooks and Scripts

The resources directory contains the list of notebooks to demonstrate Cerebrum, and
standalone batch scripts to build/simulate using cerbrum.

## IPython Notebooks

| Model                                     | Notebook                                                          |
| ----------------------------------------- | ----------------------------------------------------------------- |
| Demo of Cerebrum V1L4 model               | [V1 L4 IPython Notebook](resources/notebooks/V1L4-Notebook.ipynb) |
| Demo of Cerebrum V1 model                 | [V1 IPython Notebook](resources/notebooks/V1-Notebook.ipynb)      |
| Demo of Cerebrum V1 model on Cybershuttle | [V1 IPython Notebook](resources/notebooks/V1-CS-Notebook.ipynb)   |
| Demo of WGN Sleep model                   | [WGN Sleep IPython Notebook](resources/notebooks/WGN-Sleep.ipynb) |

## Command-line scripts

| Model                              | Scripts                                              |
| ---------------------------------- | ---------------------------------------------------- |
| Build/Simulate Cerebrum V1 model   | [V1 script](resources/notebooks/cerebrumv1.py)       |
| Simulate Cerebrum V1 model w. BMTK | [V1 script](resources/notebooks/v1_bmtk_simulate.py) |
| Simulate Cerebrum V1 model w. NEST | [V1 script](resources/notebooks/v1_nest_simulate.py) |

# Install Airavata Cerebrum

Airavata Cerebrum requires python3.10+ environment.
It is currently tested only in Linux operating system.

To install locally, we recommend to create a virtual environment using
conda ([miniforge](https://github.com/conda-forge/miniforge) for a faster
installation) as below:

```
conda config --add channels conda-forge
conda create -n cerebrum python=3.10 nest-simulator mpi4py nodejs
conda activate cerebrum
```

nest-simulator should be installed when creating the conda environment
since there is no PyPI package available for NEST.

Cerebrum depends upon NEST and BMTK, both of which depend upon mpi4py, the python
interface to MPI. In the above commands, we attempt to install via conda.
However, in some cases where conda's MPI causes some errors, we recommend
that MPI is installed from the OS.
In case of Ubuntu, this can be accomplished by

```
sudo apt install openmpi-bin  libopenmpi-dev
```

After installing MPI libraries, mpi4py can be installed via pip manually.

To install Airavata Cerebrum into the environment created above,
pip can be used with the git option as follows.

```
pip install git+https://github.com/apache/airavata-cerebrum.git
```

See [INSTALL.md](INSTALL.md) for details about how to install for a
development environment and other issues.
