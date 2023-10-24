# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Classes and other infrastructure for creating datasets.

For concrete instances of datasets see ``datasets.py``.

Code adapted from:
https://github.com/hughsalimbeni/bayesian_benchmarks/blob/master/bayesian_benchmarks/data.py
"""
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import AbstractSet, Callable, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import scipy
from check_shapes import check_shapes

from benchmark.registry import TaggedRegistry
from benchmark.tag import Tag, TagReq
from gpflow.base import AnyNDArray


class DatasetTag(Tag["DatasetTag"]):
    """
    A tag that can be applied to a dataset.
    """


DatasetReq = TagReq[DatasetTag]

REGRESSION = DatasetTag("REGRESSION")
CLASSIFICATION = DatasetTag("CLASSIFICATION")
TINY = DatasetTag("TINY")
MEDIUM = DatasetTag("MEDIUM")
LARGE = DatasetTag("LARGE")
SYNTHETIC = DatasetTag("SYNTHETIC")
REAL_DATA = DatasetTag("REAL_DATA")


@dataclass
class XYData:
    """ Data with corresponding X and Y points. """

    tags: AbstractSet[DatasetTag]
    """ Tags signalling the properties of this data. """

    X: AnyNDArray
    """ Inputs. """

    Y: AnyNDArray
    """ Outputs. """

    @check_shapes(
        "self.X: [N, D]",
        "self.Y: [N, P]",
    )
    def __post_init__(self) -> None:
        pass

    @property
    def N(self) -> int:
        """ Number of rows of data. """
        return self.X.shape[0]  # type: ignore[no-any-return]

    @property
    def D(self) -> int:
        """ Number of inputs. """
        return self.X.shape[1]  # type: ignore[no-any-return]

    @property
    def P(self) -> int:
        """ Number of outputs. """
        return self.Y.shape[1]  # type: ignore[no-any-return]

    @property
    def XY(self) -> Tuple[AnyNDArray, AnyNDArray]:
        """ This data as a (X, Y) tuple. """
        return (self.X, self.Y)


@dataclass
class Dataset:
    """ A set of both training and testing data. """

    name: str
    """ Unique name of this dataset. """

    tags: AbstractSet[DatasetTag]
    """ Tags signalling the properties of this data. """

    train: XYData
    """ Training data. """

    test: XYData
    """ Test data. """

    @check_shapes(
        "self.train.X: [N_train, D]",
        "self.train.Y: [N_train, P]",
        "self.test.X: [N_test, D]",
        "self.test.Y: [N_test, P]",
    )
    def __post_init__(self) -> None:
        assert self.name.isidentifier()
        assert self.tags == self.train.tags
        assert self.tags == self.test.tags

    @property
    def D(self) -> int:
        """ Number of inputs. """
        return self.train.D

    @property
    def P(self) -> int:
        """ Number of outputs. """
        return self.train.P

    @property
    def stats(self) -> pd.DataFrame:
        """
        Get some basic statistics about this dataset; for debugging, logging, etc.
        """
        return pd.DataFrame(
            [
                ("N_train", self.train.N),
                ("N_test", self.test.N),
                ("D", self.D),
                ("P", self.P),
            ],
            columns=["statistic", "value"],
        )


class DatasetFactory(ABC):
    """
    A way to generate a dataset.

    The factory itself should be cheap to create, though it may take some time to actually generate
    the dataset.
    """

    name: str
    """
    Unique name of this dataset / dataset factory.
    """

    tags: AbstractSet[DatasetTag]
    """
    Tags representing properties / capabilities of this dataset.
    """

    @abstractmethod
    def create_dataset(self, cache_dir: Path) -> Dataset:
        """
        Create the dataset.

        This method should be deterministic - if called multiple times it should return the same
        data. If you need randomness, please use a fixed seed.

        :param cache_dir: A directory you can use to read/write any cache data.
        """


DATASET_FACTORIES: TaggedRegistry[DatasetFactory, DatasetTag] = TaggedRegistry()


DatasetFactoryFn = Callable[[Path], Dataset]
"""
A function that can be used as a :class:`ModelFactory`.

Takes a cache directory, and produces a :class:`Dataset`.
"""


class FnDatasetFactory(DatasetFactory):
    """
    Adapter from a function to a :class:`DatasetFactory`.
    """

    def __init__(self, name: str, tags: AbstractSet[DatasetTag], fn: DatasetFactoryFn) -> None:
        self.name = name
        self.tags = tags
        self._fn = fn

    def create_dataset(self, cache_dir: Path) -> Dataset:
        return self._fn(cache_dir)


def make_dataset_factory(
    tags: AbstractSet[DatasetTag],
) -> Callable[[DatasetFactoryFn], FnDatasetFactory]:
    """
    Decorator for turning a function into a :class:`DatasetFactory`.
    """

    def wrap(fn: DatasetFactoryFn) -> FnDatasetFactory:
        name = fn.__name__
        factory = FnDatasetFactory(name, tags, fn)
        DATASET_FACTORIES.add(factory)
        return factory

    return wrap


def getNrOutlierAndRndOutlierIds(n, OUTLIER_RATIO, rng):
    nrOutliers = int(n * OUTLIER_RATIO)
    trueOutlierIds = np.arange(n)
    rng.shuffle(trueOutlierIds)
    trueOutlierIds = trueOutlierIds[0:nrOutliers]

    trueOutlierIds_zeroOne = np.zeros(n, dtype=np.int_)
    trueOutlierIds_zeroOne[trueOutlierIds] = 1
    return nrOutliers, trueOutlierIds, trueOutlierIds_zeroOne


def addUniformNoise(X, Y, OUTLIER_RATIO, symmetric, rng):
    nrOutliers, trueOutlierIds, _ = getNrOutlierAndRndOutlierIds(Y.shape[0], OUTLIER_RATIO, rng)
    if nrOutliers == 0:
        return Y

    y_std = np.std(Y)
    CUT_OFFSET = 3.0 * y_std
    OUTLIER_LENGTH = 12.0 * y_std

    trueOutlierSamplesRaw = rng.uniform(low=0.0, high=OUTLIER_LENGTH, size=nrOutliers)

    if symmetric:
        lowerOutliers = -trueOutlierSamplesRaw[trueOutlierSamplesRaw < OUTLIER_LENGTH/2] - CUT_OFFSET 
        higherOutliers = trueOutlierSamplesRaw[trueOutlierSamplesRaw >= OUTLIER_LENGTH/2] - (OUTLIER_LENGTH/2) + CUT_OFFSET
    else:
        lowerOutliers = -trueOutlierSamplesRaw * 0.5 - CUT_OFFSET
        higherOutliers = []
        if rng.uniform() > 0.5:
            # swap
            higherOutliers = -1.0 * lowerOutliers
            lowerOutliers = []
    noise = np.hstack((lowerOutliers, higherOutliers))
    Y[trueOutlierIds] += noise.reshape(-1, 1)
    return X, Y


def addFocused(X, Y, TRAINING_DATA_OUTLIER_RATIO, rng):

    nrOutliers, trueOutlierIds, _ = getNrOutlierAndRndOutlierIds(Y.shape[0], TRAINING_DATA_OUTLIER_RATIO, rng)

    d = X.shape[1]

    midPoint = int(X.shape[0] / 2)

    FOCUS_POINTS_IDS = np.argsort(X, axis=0)[midPoint, :]

    CONCENTRATION_X = scipy.stats.median_abs_deviation(X, axis=0) * 0.1 * d
    jitterX = CONCENTRATION_X * (rng.uniform(low=0.0, high=1.0, size=(nrOutliers, d)) - 0.5)

    CONCENTRATION_Y = scipy.stats.median_abs_deviation(Y) * 0.1
    OFFSET_Y = 3.0 * np.std(Y)
    jitterY = CONCENTRATION_Y * rng.uniform(low=0.0, high=1.0, size=nrOutliers)

    X[trueOutlierIds, :] = X[FOCUS_POINTS_IDS, np.arange(d)] + jitterX
    Y[trueOutlierIds] = (np.median(Y[FOCUS_POINTS_IDS]) - OFFSET_Y - jitterY).reshape(-1,1)

    return X, Y


@check_shapes(
    "X: [N, D]",
    "return: [N, D]",
)
def _normalise(X: AnyNDArray) -> AnyNDArray:
    X_mean = np.average(X, axis=0)[None, :]
    X_std = 1e-6 + np.std(X, axis=0)[None, :]
    return (X - X_mean) / X_std  # type: ignore[no-any-return]


@check_shapes(
    "X: [N, D]",
    "Y: [N, P]",
)
def _create_dataset(
    name: str,
    tags: AbstractSet[DatasetTag],
    test_fraction: float,
    normalise: bool,
    outliers: str,
    outliers_ratio: float,
    X: AnyNDArray,
    Y: AnyNDArray,
) -> Dataset:
    """
    Randomly splits raw data into a training and testing and create a new :class:`Dataset` from it.

    :param name: Name of the resulting `Dataset`.
    :param tags: Tags of the resulting `Dataset`.
    :param test_fraction: Fraction of data to use for the test set. The rest is used for training.
    :param normalise: Whether to normalise the data to mean 0 and variance 1.
    :param X: Input values.
    :param Y: Output values.
    """
    if normalise:
        X = _normalise(X)
        Y = _normalise(Y)
    N = X.shape[0]
    rng = np.random.default_rng(20220722)
    indices = rng.permutation(N)
    N_test = round(test_fraction * N)
    train_indices = indices[N_test:]
    test_indices = indices[:N_test]
    if outliers == 'focused':
        X[train_indices], Y[train_indices] = addFocused(X[train_indices], Y[train_indices], outliers_ratio, rng=rng)
    if outliers == 'asymmetric':
        X[train_indices], Y[train_indices] = addUniformNoise(X[train_indices], Y[train_indices], outliers_ratio, symmetric=False, rng=rng)
    if outliers == 'uniform':
        X[train_indices], Y[train_indices] = addUniformNoise(X[train_indices], Y[train_indices], outliers_ratio, symmetric=True, rng=rng)
    train = XYData(tags, X=X[train_indices], Y=Y[train_indices])
    test = XYData(tags, X=X[test_indices], Y=Y[test_indices])
    return Dataset(name=name, tags=tags, train=train, test=test)


def _get_data_file_from_url(cache_dir: Path, url: str, archive_member: Optional[Path]) -> Path:
    """
    Downloads a URL to a file, and returns the path to it.

    This will cache the downloaded file, and return the cached result if present.

    :param cache_dir: Where to read / write downloaded data.
    :param url: URL to download.
    :param archive_member: If not `None` the URL is assumed to point to a zip-file, which will be
        un-zipped, and the member with this relative path will be returned.
        If `None` the downloaded file will be returned without unzipping.
    """
    file_name = Path(urlparse(url).path).name
    file_path = cache_dir / file_name
    if not file_path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(url) as response:
            data = response.read()
        file_path.write_bytes(data)
        if archive_member:
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extract(str(archive_member), cache_dir)
    if archive_member:
        return cache_dir / archive_member
    return file_path


UnsplitArrayFactory = Callable[[np.random.Generator], Tuple[AnyNDArray, AnyNDArray]]
"""
Generate synthetic data and return it as a `(X, Y)` tuple.

The returned data is assumed to not yet have been split into training/test data.
"""


def make_unsplit_array_dataset_factory(
    tags: AbstractSet[DatasetTag],
    test_fraction: float,
    normalise: bool,
    outliers=None,
    outliers_ratio: float = 0.1
) -> Callable[[UnsplitArrayFactory], FnDatasetFactory]:
    """
    Decorator for creating a :class:`DatasetFactory` that generates synthetic data.

    The generated data is assumed to not yet have been split into training/test data.
    """

    def wrap(array_factory: UnsplitArrayFactory) -> FnDatasetFactory:
        name = array_factory.__name__

        @wraps(array_factory)
        def wrapper(cache_dir: Path) -> Dataset:
            rng = np.random.default_rng(20220808)
            X, Y = array_factory(rng)
            return _create_dataset(name, tags, test_fraction, normalise, outliers, outliers_ratio, X, Y)

        return make_dataset_factory(tags)(wrapper)

    return wrap


DataReader = Callable[[Path], Tuple[AnyNDArray, AnyNDArray]]
"""
A function that reads data from a file.

Read data from the given path and return it as a `(X, Y)` tuple.
"""


def make_url_dataset_factory(
    tags: AbstractSet[DatasetTag],
    url: str,
    archive_member: Optional[Path] = None,
    test_fraction: float = 0.1,
    normalise: bool = True,
    outliers=None,
    outliers_ratio: float = 0.1
) -> Callable[[DataReader], FnDatasetFactory]:
    """
    Decorator for creating a :class:`DatasetFactory` that downloads data from a URL.

    :param tags: Tags that applies to the data.
    :param url: URL to download.
    :param archive_member: If not `None` the URL is assumed to point to a zip-file, which will be
        un-zipped, and the member with this relative path will be used.
        If `None` the downloaded file will be used without unzipping.
    :param test_fraction: Fraction of data to use for the test set. The rest is used for training.
    :param normalise: Whether to normalise the data to mean 0 and variance 1.
    """

    def wrap(data_reader: DataReader) -> FnDatasetFactory:
        name = data_reader.__name__

        @wraps(data_reader)
        def wrapper(cache_dir: Path) -> Dataset:
            dataset_dir = cache_dir / name
            data_path = _get_data_file_from_url(dataset_dir, url, archive_member)
            X, Y = data_reader(data_path)
            return _create_dataset(name, tags, test_fraction, normalise, outliers, outliers_ratio, X, Y)

        return make_dataset_factory(tags)(wrapper)

    return wrap