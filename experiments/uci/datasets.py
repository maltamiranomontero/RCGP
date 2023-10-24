from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import gpflow

from experiments.uci.dataset_api import (
    LARGE,
    MEDIUM,
    REAL_DATA,
    REGRESSION,
    SYNTHETIC,
    TINY,
    make_unsplit_array_dataset_factory,
    make_url_dataset_factory,
)
from gpflow.base import AnyNDArray


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, MEDIUM, SYNTHETIC},
    test_fraction=0.1,
    normalise=False,
)
def synthetic(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    np.random.seed(321)
    n_points = 300
    sigma_n = 0.1
    lengthscale = 1
    variance = 1

    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale,
                                               variance=variance)

    X = np.linspace(0, 20, n_points).reshape(n_points, 1)
    F = np.random.multivariate_normal(mean=np.zeros(len(X)),
                                      cov=kernel(X, X)).reshape(n_points, 1)

    Y = F + np.random.normal(scale=sigma_n,
                             size=n_points).reshape(n_points, 1)
    return X, Y


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "housing/housing.data",
)
def boston(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "concrete/compressive/Concrete_Data.xls",
)
def concrete(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path).values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "00242/ENB2012_data.xlsx",
)
def energy(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path, engine="openpyxl", usecols=np.arange(9)).dropna().values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "/00243/yacht_hydrodynamics.data",
)
def yacht(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 6, N = 308
    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1].reshape(-1, 1)

@make_unsplit_array_dataset_factory(
    tags={REGRESSION, MEDIUM, SYNTHETIC},
    test_fraction=0.1,
    normalise=False,
    outliers="uniform",
    outliers_ratio=0.1
)
def synthetic_uniform(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    np.random.seed(321)
    n_points = 300
    sigma_n = 0.1
    lengthscale = 1
    variance = 1

    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale,
                                               variance=variance)

    X = np.linspace(0, 20, n_points).reshape(n_points, 1)
    F = np.random.multivariate_normal(mean=np.zeros(len(X)),
                                      cov=kernel(X, X)).reshape(n_points, 1)

    Y = F + np.random.normal(scale=sigma_n,
                             size=n_points).reshape(n_points, 1)
    return X, Y


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "housing/housing.data",
    outliers="uniform",
    outliers_ratio=0.1
)
def boston_uniform(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "concrete/compressive/Concrete_Data.xls",
    outliers="uniform",
    outliers_ratio=0.1
)
def concrete_uniform(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "00242/ENB2012_data.xlsx",
    outliers="uniform",
    outliers_ratio=0.1
)
def energy_uniform(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path, engine="openpyxl", usecols=np.arange(9)).dropna().values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "/00243/yacht_hydrodynamics.data",
    outliers="uniform",
    outliers_ratio=0.1
)
def yacht_uniform(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 6, N = 308
    data = np.loadtxt(path)
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, MEDIUM, SYNTHETIC},
    test_fraction=0.1,
    normalise=False,
    outliers="asymmetric",
    outliers_ratio=0.2
)
def synthetic_asymmetric(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    np.random.seed(321)
    n_points = 300
    sigma_n = 0.1
    lengthscale = 1
    variance = 1

    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale,
                                               variance=variance)

    X = np.linspace(0, 20, n_points).reshape(n_points, 1)
    F = np.random.multivariate_normal(mean=np.zeros(len(X)),
                                      cov=kernel(X, X)).reshape(n_points, 1)

    Y = F + np.random.normal(scale=sigma_n,
                             size=n_points).reshape(n_points, 1)
    return X, Y


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "housing/housing.data",
    outliers="asymmetric",
    outliers_ratio=0.1
)
def boston_asymmetric(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "concrete/compressive/Concrete_Data.xls",
    outliers="asymmetric",
    outliers_ratio=0.1
)
def concrete_asymmetric(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "00242/ENB2012_data.xlsx",
    outliers="asymmetric",
    outliers_ratio=0.1
)
def energy_asymmetric(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path, engine="openpyxl", usecols=np.arange(9)).dropna().values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "/00243/yacht_hydrodynamics.data",
    outliers="asymmetric",
    outliers_ratio=0.1
)
def yacht_asymmetric(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 6, N = 308
    data = np.loadtxt(path)
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, MEDIUM, SYNTHETIC},
    test_fraction=0.1,
    normalise=False,
    outliers="focused",
    outliers_ratio=0.1
)
def synthetic_focused(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    np.random.seed(321)
    n_points = 300
    sigma_n = 0.1
    lengthscale = 1
    variance = 1

    kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale,
                                               variance=variance)

    X = np.linspace(0, 20, n_points).reshape(n_points, 1)
    F = np.random.multivariate_normal(mean=np.zeros(len(X)),
                                      cov=kernel(X, X)).reshape(n_points, 1)

    Y = F + np.random.normal(scale=sigma_n,
                             size=n_points).reshape(n_points, 1)
    return X, Y


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "housing/housing.data",
    outliers="focused",
    outliers_ratio=0.1
)
def boston_focused(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "concrete/compressive/Concrete_Data.xls",
    outliers="focused",
    outliers_ratio=0.1
)
def concrete_focused(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path).values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "00242/ENB2012_data.xlsx",
    outliers="focused",
    outliers_ratio=0.1
)
def energy_focused(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 10, N = 1030
    data = pd.read_excel(path, engine="openpyxl", usecols=np.arange(9)).dropna().values
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "/00243/yacht_hydrodynamics.data",
    outliers="focused",
    outliers_ratio=0.1
)
def yacht_focused(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    # D = 6, N = 308
    data = np.loadtxt(path)
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)
    return X, Y
