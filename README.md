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
1. A major technical limitation of the analysis performed here is that neither trend nor seasonality have been removed from the timeseries for each store-item combination from the data. Tree-based methods cannot extrapolate and so can't handle trend ([1](http://freerangestats.info/blog/2016/12/10/extrapolation), [2](https://srome.github.io/Dealing-With-Trends-Combine-a-Random-Walk-with-a-Tree-Based-Model-to-Predict-Time-Series-Data/)). The timeseries should have been transformed to remove the underlying signal before attempting to use ML-based techniques for timeseries forecasting. While [Deep Learning (DL)](https://en.wikipedia.org/wiki/Deep_learning) doesn't have this limitation (and was not used here), DL approaches such as [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) perform better at forecasting if the underlying structure has been removed from the time-series ([1](https://www.quora.com/Why-are-the-data-used-in-LSTM-needed-to-be-transformed-into-stationary-when-processing-time-series-It-seems-like-the-process-of-backpropagation-is-curve-fitting/answer/Marco-Santanch%C3%A9), [2](https://www.quora.com/Can-an-LSTM-predict-the-time-series-if-they-are-not-stationary/answer/Nowan-Ilfideme)). Again, [structure should be removed](https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd-1f) before trying to use deep learning for forecasting.
2. A related technical complication with using ML for non-stationary timeseries forecasting is that the stationarity transformations need to be reverted to the original scale, **without using the original data**. If a ML model is developed using a training split (which has been transformed) of the overall dataset, and used to forecast an out-of-sample (test) split, then the forecast will be on the transformed scale and it needs to be inverted **without using the true values of the test split** ([1](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/#comment-486543)). The framework developed in this project does not make an allocation for implementing such invertibility. The above-mentioned [tutorial](https://srome.github.io/Dealing-With-Trends-Combine-a-Random-Walk-with-a-Tree-Based-Model-to-Predict-Time-Series-Data/) shows how to invert a transformation when the test set data is used to perform the inversion. However, when generating a forecast in a real-life scenario, the test set (i.e. the real values for the period covered by the forecast) are not known. For example, when forecasting 14 days into the future, the true values of those 14 days will not be available to us and we will have to wait for 14 days in order to evaluate the forecast. If that is the case, we **also will have to wait for 14 days to have access to the true values** in order to use them to invert the forecasted values - obviously, this is not practical since we need the forecasted values (on the correct scale) as soon as they are available. For this reason, we need to pick transformations that can be reverted without using the test set (true values) - this means we must use the training data to revert the forecasted values back to the original scale.

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
