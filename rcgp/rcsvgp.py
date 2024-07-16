from typing import Optional

import tensorflow as tf
import numpy as np
from check_shapes import inherit_check_shapes


from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingPoints
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper, InducingPointsLike
from gpflow.config import default_jitter
from gpflow.utilities import to_default_float


class RCSGPR(GPModel, InternalDataTrainingLossMixin):
    """
    Robust and Conjugate Sparse Gaussian Process Regression
    This method only works with a Gaussian likelihood, its variance is
    initialized to `noise_variance`.
    """
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPointsLike,
        weighting_function,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = Gaussian(noise_variance)
        X_data, Y_data = data_input_to_tensor(data)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable: InducingPoints = inducingpoint_wrapper(inducing_variable)

        self.weighting_function = weighting_function

    def elbo(self) -> tf.Tensor:

        X, Y = self.data
        K = self.kernel(X, full_cov=False)

        W = self.weighting_function.W(X, Y)
        W_dy = self.weighting_function.dy(X, Y)

        num_inducing = self.inducing_variable.num_inducing

        sigma_sq = self.likelihood.variance_at(X)
        sigma = tf.sqrt(sigma_sq)
        W_sigma = tf.squeeze(W/sigma, axis=-1)

        kuf = Kuf(self.inducing_variable, self.kernel, X)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)

        Lkuf = tf.linalg.triangular_solve(L, kuf, lower=True)

        nu = (sigma_sq**-1)*Y*(W**2)-2*W*W_dy

        MKuf = kuf*W_sigma

        A = kuu + tf.linalg.matmul(MKuf, MKuf, transpose_b=True)

        LA = tf.linalg.cholesky(A)

        kufnu = tf.linalg.matmul(kuf, nu)

        tmp1 = tf.linalg.triangular_solve(LA, kufnu, lower=True)

        quad = tf.reduce_sum(tmp1**2)/2

        C1 = tf.reduce_sum(Y*Y*(W**2)*self.likelihood.variance_at(X)**-1)/2
        C2 = - tf.reduce_sum(Y*2*W*W_dy)
        C3 = - tf.reduce_sum(W**2)

        C = tf.reduce_sum(C1+C2+C3)

        S = K - tf.reduce_sum(Lkuf**2, axis=0)

        trace = tf.reduce_sum(S*W_sigma**2)

        const = 0.5*to_default_float(num_inducing)*np.log(2*np.pi)

        det1 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LA)))
        det2 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        return (quad - C - trace - det1 + det2 + const)

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:

        X_data, Y_data = self.data

        err = Y_data - self.mean_function(X_data)

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)

        W = self.weighting_function.W(X_data, err)
        W_dy = self.weighting_function.dy(X_data, err)

        diag_W = tf.linalg.diag(tf.squeeze(W, axis=-1))

        sigma_sq = self.likelihood.variance_at(X_data)
        sigma = tf.sqrt(sigma_sq)

        Yt = (sigma_sq**-1)*err*(W**2)-2*W*W_dy

        WKuf = tf.linalg.matmul(kuf, diag_W/sigma)

        L = tf.linalg.cholesky(kuu)

        A = kuu + tf.linalg.matmul(WKuf, WKuf, transpose_b=True)

        LA = tf.linalg.cholesky(A) 

        tmp1 = tf.linalg.triangular_solve(LA, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LA, kuf, lower=True)

        tmp3 = tf.linalg.matmul(tmp1, tmp2, transpose_a=True)

        tmp4 = tf.linalg.triangular_solve(L, Kus)

        mean = tf.linalg.matmul(tmp3, Yt)

        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
                - tf.linalg.matmul(tmp4, tmp4, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp1), 0)
                - tf.reduce_sum(tf.square(tmp4), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var
