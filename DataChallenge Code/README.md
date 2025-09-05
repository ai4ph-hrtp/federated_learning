# AI4PH: Federated Learning Workshop

## Setup

The first step is to install the necessary R packages. To do this, run `install_packages.R`.

## Federated learning in R tutorial

This workshop has two main documents:

- `fl_tutorial.qmd`/`fl_tutorial.html`: Main tutorial, can be run interactively (`.qmd`) or read as a static document (`.html`).
- `fl_reference.qmd`/`fl_reference.html`: Reference document for the main `fed_learn()` function, explaining the necessary data preparation steps and the function's inputs and outputs. Meant to be read as a static document (`.html`) and **not** run interactively.

These documents are supported by the following files:

- `fl_funs.R`: Contains the federated learning functions used in this tutorial, including `fed_learn()`. There should be no reason to reference this file directly unless something breaks unexpectedly. **However, you will have to source it in your R script (`source("fl_funs.R")`) to use the functions.**
- `sim/`: The directory containing the simulated data used in `fl_tutorial.qmd`.

## Federated learning in Python

To encourage further exploration and learning beyond the workshop, we have included instructions for running an example of a simulation of federated learning in Flower, a federated learning framework in Python, [taken from the official Flower GitHub repository](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist). These commands, which should be run from the root directory of this project, require an installation of Git and Python:

```{bash}
# Download files
git clone --depth=1 https://github.com/adap/flower.git _tmp \
		&& mv _tmp/examples/sklearn-logreg-mnist . \
		&& rm -rf _tmp && cd sklearn-logreg-mnist

# Create virtual environment and install packages
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run the simulation (will download required dataset)
flwr run .

# Run the simulation overriding the options in `pyproject.toml`
flwr run . --run-config "num-server-rounds=5 fraction-fit=0.25"
```
