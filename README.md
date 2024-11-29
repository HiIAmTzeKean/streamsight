# Streamsight

![streamsight-logo](docs/assets/logo_removebg.png)

Streamsight is an offline Reccomender Systems (RecSys) evaluation toolkit that respects a global timeline.
The aim is to partition the data into different windows where data is incrementally released for the programmer
to fit, train and submit predictions. This aims to provide a close simulation of an online setting when evaluating
RecSys algorithms.

## Full Flow Structure

![full-flow](docs/assets/full_flow.png)

[![PyPI Latest Release](https://img.shields.io/pypi/v/streamsight.svg)](https://pypi.org/project/streamsight/)&nbsp;&nbsp;
[![Docs](https://github.com/hiiamtzekean/streamsight/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/hiiamtzekean/streamsight/)&nbsp;&nbsp;
[![Python version](https://img.shields.io/badge/python-3.12.5-blue)](https://www.python.org/downloads/)

## Getting Started

Clone the repository

```bash
git clone https://github.com/hiiamtzekean/streamsight
cd streamsight
```

Dependencies can be installed with `uv` for ease of management.

```bash
uv sync
```

Alternatively, you may install dependencies locally with pip and venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

The dependencies are listed in `pyproject.toml`.

## Contributing

- We welcome all contributors, be it reporting an [issue](https://github.com/hiiamtzekean/streamsight/issues),
or raising a [pull request](https://github.com/hiiamtzekean/streamsight/pulls) to fix an issue.
- When you make changes, rerun `pip install .` to test your changes.

## Documentation

The documentation can be found [here](https://hiiamtzekean.github.io/streamsight/)
and [repository](https://github.com/hiiamtzekean/streamsight) on Github.

## Citation

If you use this library in any part of your work, please cite the following papers:

```text
Ng, T. K. (2024). Streamsight: a toolkit for offline evaluation of recommender systems. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/181114
```
