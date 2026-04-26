# Model Notebooks and Scripts

This directory contains the list of notebooks to demonstrate Cerebrum, and
standalone batch scripts to build/simulate using cerbrum.

## IPython Notebooks

| Model                                     | Notebook                                      |
| ----------------------------------------- | --------------------------------------------- |
| Demo of Cerebrum V1L4 model               | [V1 L4 IPython Notebook](V1L4-Notebook.ipynb) |
| Demo of Cerebrum V1 model                 | [V1 IPython Notebook](V1-Notebook.ipynb)      |
| Demo of Cerebrum V1 model on Cybershuttle | [V1 IPython Notebook](V1-CS-Notebook.ipynb)   |
| Demo of WGN Sleep model                   | [WGN Sleep IPython Notebook](WGN-Sleep.ipynb) |

## Command-line scripts

| Model                              | Scripts                               |
| ---------------------------------- | ------------------------------------- |
| Build/Simulate Cerebrum V1 model   | [V1 script](mousev1/cli.py)           |
| Simulate Cerebrum V1 model w. BMTK | [V1 script](mousev1/simulate_cli.py)  |
| Simulate Cerebrum V1 model w. NEST | [V1 script](nest_simulate_cli.py)     |

# Install Additional Dependencies for Example Notebooks

V1/V1 L4 notebooks depend upon NEST and BMTK, both of which inturn 
depend upon mpi4py, the python interface to MPI. We can install all
these dependencies via conda. Since, in some cases, MPI can cause some errors, 
we recommend that MPI is installed using the OS package managers or using 
external package managers such as `spack`.
In case of Ubuntu, this can be accomplished by

```
sudo apt install openmpi-bin  libopenmpi-dev
```

After installing MPI libraries, all the additional dependcies can be
installed form `environment.yml` as follows:

```
conda env update -n cerebrum --file environment.yml
```

## Potential Installation Issues

### MPI Depenency Issue

- NEST and BMTK depends upon MPI -- specifically the python mpi4py package.
- Latest version of mpi4py package is only available in the PyPI and hence need to be
  installed via pip command.
- In some cases, the mpi package from conda causes failures during installation
  of mpi4py. This is due to the compilation errors arising from building the 
  C/C++ modules of mpi4py (compiled using mpicc).
- To avoid this, it is recommend to install the MPI libraries on the system first
  before creating the conda environment and installing the python libraries.
