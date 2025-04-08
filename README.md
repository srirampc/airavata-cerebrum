# Apache Airavata Cerebrum, an Integrated Neuroscience Computational Framework

## Introduction

Welcome to the Apache Airavata Cerebrum repository for the Integrated Neuroscience Computational Framework. This project aims to revolutionize how we understand and model the human brain by integrating cellular-level brain atlases with advanced computational tools. Our goal is to create a cohesive, open-source framework that allows for the seamless application of existing tools within a streamlined, lightweight environment.

## Features

- **Integration of Brain Atlases**: Merging publicly available cellular-level brain atlases into a single, accessible platform.
- **Comprehensive Modeling Tools**: Incorporating computational tools designed for modeling the entire brain.
- **Open-Source Framework**: Developing a user-friendly, open-source environment for neuroscience research.
- **Streamlined Environment**: Ensuring a lightweight, efficient framework for both beginners and advanced users.

# Try Airavata Cerebrum
Airavata Cerebrum requires python3.10+ environment.
It is currently tested only in Linux operating system.

To install locally, we recommend to create a virtual environment using
conda ([miniforge](https://github.com/conda-forge/miniforge) for a faster
installation) as below:
```
conda config --add channels conda-forge
conda create -n cerebrum python=3.10 nest-simulator mpi4py
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

Check the `resources/notebooks/V1L4-Notebook.ipynb` notebook for an example usage.

See [INSTALL.md](INSTALL.md) for details about how to install for a 
development environment and other issues.
