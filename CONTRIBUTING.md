# Contibuting to Cerebrum

## Setting up Development Environment

### Clone the source github repo
`airavata-cerebrum` includes three submodules:
[abc_atlas_access](https://github.com/srirampc/abc_atlas_access),
[aisynphys](https://github.com/srirampc/aisynphys) and
[codetiming](https://github.com/srirampc/codetiming).
Clone the repository along with all submodules as below.
```sh
git clone --recurse-submodules https://github.com/apache/airavata-cerebrum.git
```

In the wheel file, all the dependencies are included with in
the `airavata_cerebrum/ext` namespace.
To get these modules to work in developement environment, create the following
links to the paths in the _ext_ directory:
```sh
cd airavata-cerebrum/airavata_cerebrum
ln -s ../../ext/codetiming/codetiming/
ln -s ../../ext/abc_atlas_access/src/abc_atlas_access/
ln -s ../../ext/aisynphys/aisynphys/
ln -s ../../ext/abc_atlas_access/src/manifest_builder/
```

### Installing Environment For Development

Development environment for Airavata Cerebrum is created as a python3.10+ 
virtual environment in `conda` using the `environment.yml` file.
```sh
conda config --add channels conda-forge
conda env create -n cbmdev -f environment.yml
conda activate cbmdev
```

`environment.yml` includes the version of each of the package to make conda's 
dependency resolution algorithm to run faster. 

### Buliding dist/wheels

We use [poetry](https://python-poetry.org/) a build tool which can be buit 
as follows.
Intall the follwing dependencies for building the dist/wheels:
```sh
pip install build
pip install twine==6.0.1
```

Build and upload to pypi:
```sh
rm -rf dist; python3 -m build
python3 -m twine upload --repository pypi dist/* --verbose
```



## Potential Environment Issues

### Miniforge `conda` Issue when installed with `spack`

If installing conda with miniforge, in some cases the following error appears:
`ModuleNotFoundError: No module named 'conda'`

This happens when the conda script (`$CONDA_EXE`)  has in its first line 
`#!/usr/bin/env python`, which picks up the python from the `cerebrum` environment
which is currently being installed instead of the base environment. To fix this,
replace the `/usr/bin/env python` with the python installed in the base 
environment (generally corresponds of the environment variable `$CONDA_PYTHON_EXE`).
