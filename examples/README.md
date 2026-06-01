# Examples of `airavata-cerebrum` applications

This folder contains jupyter notebooks, marimo notebooks and 
other standalone scripts demonstating the use of
[airavata-cerebrum](https://github.com/apache/airavata-cerebrum) 
software in building the neuroscience models. 

## Contents

1. [Installation](#install-airavata-cerebrum-and-dependencies)
2. [V1 Model](#v1-model)
3. [Sonata Editor](#sonata-edge-editor)
4. [WGN Sleep Model](#wgn-sleep-model)
5. [Common Installation Issues](#installation-issues)
## Install airavata-cerebrum and dependencies

V1/V1 L4 notebooks listed below depend upon 
[airavata-cerebrum](https://github.com/apache/airavata-cerebrum), 
[NEST](https://www.nest-simulator.org/) ,
[BMTK](https://alleninstitute.github.io/bmtk/) and other dependencies.
BMTK and NEST inturn depend upon [mpi4py](https://mpi4py.readthedocs.io/),
the python interface to MPI. 
All these dependencies can be installed via conda using the given
`environment.yml` (in this directory) as follows:

```sh
conda env create -n arv_cbm -f environment.yml
conda activate arv_cbm
```

See [below](#installation-issues) for common installation issues.


## V1 Model

The following notebooks demonstrate the use of airavata-cerebrum to gather 
data from Brain Atlases and other databases,  build the model and 
simulate using NEST simulator. See the [v1/README.md](v1/README.md) 
for a detail discussion on each phase of model construction.

### IPython Notebooks

| Model                             | Notebook                                         |
| --------------------------------- | ------------------------------------------------ |
| Complete V1 model                 | [V1 IPython Notebook](v1/V1-Notebook.ipynb)      |
| Complete V1 model w. Cybershuttle | [V1 IPython Notebook](v1/V1-CS-Notebook.ipynb)   |
| V1 model restricted to L4         | [V1 L4 IPython Notebook](v1/V1L4-Notebook.ipynb) |

### marimo Notebooks

| Model                      | Notebook                                     |
| -------------------------- | -------------------------------------------- |
| Complete V1 model          | [V1 marimo Notebook](v1/marimo_v1.py)        |
| Data views w.r.t V1        | [V1 db marimo Notebook](v1/marimo_v1l4db.py) |
| V1 model  resricted to L4  | [V1 L4 marimo Notebook](v1/marimo_v1l4.py)   |



### Command-line scripts

The following standalone scripts can be used to build/simulate V1 models 
via command line or as a batch run.


| Model                              | Scripts                                       |
| ---------------------------------- | --------------------------------------------- |
| Build/Simulate Cerebrum V1 model   | [V1 script](v1/src/cli.py)                    |
| Simulate Cerebrum V1 model w. BMTK | [V1 BMTK script](v1/src/simulate_cli.py)      |
| Simulate Cerebrum V1 model w. NEST | [V1 NEST script](v1/src/nest_simulate_cli.py) |

## SONATA Edge Editor


These notebooks show a demo of edititng edges in a SONATA file: A specific 
case of selecting all the edges of a given edge type id and
replacing it with a random subset of edges. 
See [editor/README.md](editor/README.md) for a description of this 
functionality.

| Model               | Notebook                                          |
| ------------------- | ------------------------------------------------- |
| jupyter Editor      | [IPython Notebook](editor/SonataEditorDemo.ipynb) |
| marimo Editor       | [marimo Notebook](editor/marimo_sonata_editor.py) |


## WGN Sleep Model

The notebook demonstrates the use of cerebrum just as an interface to the 
model builder, with user's own custom data and builders.

| Model                 | Notebook                                  |
| --------------------- | ----------------------------------------- |
| WGN Sleep model       | [IPython Notebook](sleep/WGN-Sleep.ipynb) |


## Installation Issues

### MPI Depenency Issues

- NEST and BMTK depends upon MPI -- specifically the python mpi4py package.
- Latest version of mpi4py package is only available in the PyPI and hence need to be
  installed via pip command.
- In some cases, the mpi package from conda causes failures during installation
  of mpi4py. This is due to the compilation errors arising from building the 
  C/C++ modules of mpi4py (compiled using mpicc).
- To avoid this, it is recommend to install the MPI libraries on the system first
  before creating the conda environment and installing the python libraries.
  MPI can be installed using via OS package managers (`apt`, `snap`, etc.) (or) 
  external package managers such as `spack`. In case of Ubuntu, this can be
  accomplished by
  ```sh
  sudo apt install openmpi-bin  libopenmpi-dev
  ```
