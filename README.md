# Streamsight

![logo](https://hiiamtzekean.github.io/Streamsight/_static/logo_removebg.png)

The purpose of this Final Year Project is to design and implement a toolkit for
evaluating Recommendation System (RecSys) which respects the temporal aspect
during the data splitting process and incrementally release data as close
to a live production setting as possible. We aim to achieve this through
provision of API for the programmer to interact with the objects in the library.

[![PyPI Latest Release](https://img.shields.io/pypi/v/streamsight.svg)](https://pypi.org/project/streamsight/)&nbsp;&nbsp;
[![Docs](https://github.com/HiIAmTzeKean/Streamsight/actions/workflows/pages/pages-build-deployment/badge.svg)](https://hiiamtzekean.github.io/Streamsight/)&nbsp;&nbsp;
[![Python version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)

## Table of Contents
- [Streamsight](#streamsight)
  - [Table of Contents](#table-of-contents)
  - [Installation with Github](#installation-with-github)
    - [Installation through poetry](#installation-through-poetry)
    - [Installation through pip](#installation-through-pip)
  - [Installation with PyPI](#installation-with-pypi)
  - [Documentation](#documentation)
  - [Report and Citation](#report-and-citation)


## Installation with Github

The package can be installed quickly with python `poetry` or the traditional `pip`
method. The recommended way of installation would be through `poetry` as it will
help install the dependencies along with the package. We assume that the repository
has already been cloned else you can run the following code on terminal before
continuing.

```shell
git clone https://github.com/HiIAmTzeKean/Streamsight.git
cd Streamsight
```

### Installation through poetry

The following code assumes that you do not have `poetry` installed yet. If you
using MacOS, you might want to consider installing `poetry` with homebrew instead.

```shell
pip install poetry
# MacOS can consider using brew install poetry
poetry install
```

### Installation through pip

The following code below assumes that you have `pip` installed and is in system
PATH.

```shell
pip install -e .
```

## Installation with PyPI

Alternatively `streamsight` is available on PyPi and can be installed through
either of the commands below. The link to PyPI can be found
[here](https://pypi.org/project/streamsight/).

```shell
# To install via pip
pip install streamsight

# To install with streamsight as a dependency
poetry add streamsight
```

## Documentation

The documentation can be found [here](https://hiiamtzekean.github.io/Streamsight/)
and [repository](https://github.com/HiIAmTzeKean/Streamsight) on Github.

## Report and Citation

The report for this project [Streamsight: a toolkit for offline evaluation of recommender systems](https://hdl.handle.net/10356/181114)
can be found in the NTU repository.

```
Ng, T. K. (2024). Streamsight: a toolkit for offline evaluation of recommender systems. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/181114
```