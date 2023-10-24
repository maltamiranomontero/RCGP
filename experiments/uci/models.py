import numpy as np
from scipy.stats.qmc import Halton  # pylint: disable=no-name-in-module

import experiments.uci.dataset_api as ds
from experiments.uci.dataset_api import XYData
from benchmark.model_api import REGRESSION, VARIATIONAL, make_model_factory
from gpflow.kernels import RBF, Kernel
from gpflow.likelihoods import Gaussian, StudentT
from gpflow.models import GPR, VGP, GPModel
from rcgp.rcgp import RCGPR
from rcgp import w


def create_rbf(data: XYData, rng: np.random.Generator) -> Kernel:
    return RBF(
        variance=rng.gamma(5.0, 0.2, []),
        lengthscales=rng.gamma(5.0, 0.2, [data.D]),
    )


def create_gaussian(data: XYData, rng: np.random.Generator) -> Gaussian:
    return Gaussian(variance=rng.gamma(5.0, 0.2, []))


def create_studentT(data: XYData, rng: np.random.Generator) -> Gaussian:
    return StudentT(scale=rng.gamma(5.0, 0.2, []), df=40)


@make_model_factory(tags={REGRESSION}, dataset_req=ds.REGRESSION & ~ds.LARGE)
def gpr(data: XYData, rng: np.random.Generator) -> GPModel:
    return GPR(
        data.XY,
        kernel=create_rbf(data, rng),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )


@make_model_factory(tags={REGRESSION, VARIATIONAL}, dataset_req=ds.REGRESSION & ~ds.LARGE)
def t_vgp(data: XYData, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.XY,
        kernel=create_rbf(data, rng),
        likelihood=create_studentT(data, rng),
    )


@make_model_factory(tags={REGRESSION}, dataset_req=ds.REGRESSION & ~ds.LARGE)
def rcgpr(data: XYData, rng: np.random.Generator) -> GPModel:
    return RCGPR(
        data.XY,
        kernel=create_rbf(data, rng),
        weighting_function=w.IMQ(C=np.quantile(np.abs(data.Y), 0.9)),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )