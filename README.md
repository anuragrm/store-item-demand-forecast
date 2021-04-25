# [Store-Item Demand Forecasting](#store-item-demand-forecasting)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edesz/store-item-demand-forecast)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edesz/store-item-demand-forecast/master/1_lgbm_trials_v2.ipynb)
![CI](https://github.com/edesz/store-item-demand-forecast/workflows/CI/badge.svg)
![CodeQL](https://github.com/edesz/store-item-demand-forecast/workflows/CodeQL/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/mit)
![OpenSource](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![prs-welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![pyup](https://pyup.io/repos/github/edesz/store-item-demand-forecast/shield.svg)

## [About](#about)

Forecasting 3 months of sales for each of 50 items across 10 stores within a chain with a [tree-based](https://en.wikipedia.org/wiki/Decision_tree_learning) [Machine Learning (ML)](https://en.wikipedia.org/wiki/Machine_learning) method ([LightGBM](https://en.wikipedia.org/wiki/LightGBM)).

## [Limitations](#limitations)
1. A major limitation of the analysis performed here is that neither trend nor seasonality have been removed from the timeseries for each store-item combination from the data ([1](http://freerangestats.info/blog/2016/12/10/extrapolation), [1](https://srome.github.io/Dealing-With-Trends-Combine-a-Random-Walk-with-a-Tree-Based-Model-to-Predict-Time-Series-Data/)). Tree-based method cannot extrapolate and so can't handle trend. The timeseries should have been transformed to remove underlying signal before attempting to use ML-based techniques for forecasting. While Deep Learning doesn't have this limitation (and was not used here), neural networks perform better at forecasting if the underlying structure has been removed from the time-series ([1](https://www.quora.com/Why-are-the-data-used-in-LSTM-needed-to-be-transformed-into-stationary-when-processing-time-series-It-seems-like-the-process-of-backpropagation-is-curve-fitting/answer/Marco-Santanch%C3%A9), [2](https://www.quora.com/Can-an-LSTM-predict-the-time-series-if-they-are-not-stationary/answer/Nowan-Ilfideme)). Again, structure should be removed before trying to use it for forecasting.

## [Project Organization](#project-organization)

    ├── LICENSE
    ├── .env                          <- environment variables (verify this is in .gitignore)
    ├── .gitignore                    <- files and folders to be ignored by version control system
    ├── .pre-commit-config.yaml       <- configuration file for pre-commit hooks
    ├── .github
    │   ├── workflows
    │       └── integrate.yml         <- configuration file for CI build on Github Actions
    │       └── codeql-analysis.yml   <- configuration file for security scanning on Github Actions
    ├── Makefile                      <- Makefile with commands like `make lint` or `make build`
    ├── README.md                     <- The top-level README for developers using this project.
    ├── environment.yml               <- configuration file to create environment to run project on Binder
    ├── data
    │   ├── raw                       <- Scripts to download or generate data
    |   └── processed                 <- merged and filtered data, sampled at daily frequency
    ├── *.ipynb                       <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                    and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
    ├── requirements.txt              <- base packages required to execute all Jupyter notebooks (incl. jupyter)
    ├── src                           <- Source code for use in this project.
    │   ├── __init__.py               <- Makes src a Python module
    │   └── *.py                      <- Scripts to use in analysis for pre-processing, visualization, training, etc.
    ├── papermill_runner.py           <- Python functions that execute system shell commands.
    └── tox.ini                       <- tox file with settings for running tox; see https://tox.readthedocs.io/en/latest/

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
