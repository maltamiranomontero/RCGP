
from typing import Optional

import tensorflow as tf
import numpy as np
from check_shapes import check_shapes, inherit_check_shapes

import gpflow

from gpflow import posteriors
from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingPoints
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper, InducingPointsLike
from gpflow.config import default_float, default_jitter
from gpflow.utilities import to_default_float

from rtgp.utils import add_likelihood_noise_cov


class RTGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})

    For a use example see :doc:`../../../../notebooks/getting_started/basic_usage`.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        diffusion_matrix,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
        self.diffusion_matrix = diffusion_matrix

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        n = tf.shape(X)[0]
        K = self.kernel(X)
        W = self.diffusion_matrix.W(X, Y)
        W_dy = self.diffusion_matrix.dy(X, Y)
        
        K_plus_sW = add_likelihood_noise_cov(K, W, self.likelihood, X)
        L_plus_sW = tf.linalg.cholesky(K_plus_sW + tf.eye(n, dtype=default_float()) * 1e-04)

        nu = (self.likelihood.variance_at(X)**-1)*Y*(W**2)-2*W*W_dy

        A = tf.linalg.triangular_solve(L_plus_sW, tf.transpose(tf.linalg.matmul(nu, K, transpose_a=True)), lower=True)
        B = tf.linalg.triangular_solve(L_plus_sW, nu*(W**-2)*self.likelihood.variance_at(X), lower=True)

        C1 = tf.matmul(Y,Y*(W**2)*self.likelihood.variance_at(X)**-1, transpose_a=True)/2
        C2 = - tf.matmul(Y,W*W_dy, transpose_a=True)
        C3 = - tf.matmul(W,W, transpose_a=True)

        C = tf.reduce_sum(C1+C2+C3)

        D1 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_plus_sW)))
        D2 = tf.reduce_sum(tf.math.log((W**2)*self.likelihood.variance_at(X)**-1))/2

        D = tf.reduce_sum(D1+D2) 

        return tf.reduce_sum(tf.linalg.matmul(A, B, transpose_a=True))/2 - C - D

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        #assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X, Y = self.data
        err = Y - self.mean_function(X)

        W = self.diffusion_matrix.W(X, err)
        W_dy = self.diffusion_matrix.dy(X, err)

        likelihood_variance = self.likelihood.variance_at(X)

        kmm = self.kernel(X)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        kmm_plus_sW = add_likelihood_noise_cov(kmm, W, self.likelihood, X)

        Lm_plus_sW = tf.linalg.cholesky(kmm_plus_sW)

        K = tf.rank(kmn)
        perm = tf.concat(
            [
                tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
                tf.reshape(0, [1]),  # [W]
                tf.reshape(K - 1, [1]),
            ],
            0,
        )  # [N]
        kmn = tf.transpose(kmn, perm)  # [..., W, N]

        leading_dims = tf.shape(kmn)[:-2]

        # Compute the projection matrix A
        Lm_plus_sW = tf.broadcast_to(Lm_plus_sW, tf.concat([leading_dims, tf.shape(Lm_plus_sW)], 0))  # [..., W, W]
        A = tf.linalg.triangular_solve(Lm_plus_sW, kmn, lower=True)  # [..., W, N]

        # compute kernel stuff
        num_func = tf.shape(err)[-1]  # R
        N = tf.shape(kmn)[-1]


        if full_cov:
            f_var = knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
            cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -3), cov_shape)  # [..., R, N, N]
        else:
            f_var = knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
            cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -2), cov_shape)  # [..., R, N]

        if not full_cov:
            f_var = tf.linalg.adjoint(f_var)  # [N, R]

        f = err - 2*likelihood_variance*W_dy/W
        B = tf.linalg.triangular_solve(Lm_plus_sW, f, lower=True)  # [..., W, N]
        f_mean_zero = tf.linalg.matmul(A, B, transpose_a=True)

        f_mean = f_mean_zero + self.mean_function(Xnew)

        return f_mean, f_var
    

class RTSGPR(GPModel, InternalDataTrainingLossMixin):
    """
    
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPointsLike,
        diffusion_matrix,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        """
        This method only works with a Gaussian likelihood, its variance is
        initialized to `noise_variance`.

        :param data: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        :param inducing_variable:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [W, D].
        :param kernel: An appropriate GPflow kernel object.
        :param mean_function: An appropriate GPflow mean function object.
        """
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
        
        self.diffusion_matrix = diffusion_matrix
    
    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        X, Y = self.data
        n = tf.shape(X)[0]
        K = self.kernel(X)

        W = self.diffusion_matrix.W(X, Y)
        W_dy = self.diffusion_matrix.dy(X, Y)
        diag_W = tf.linalg.diag(tf.squeeze(W, axis=-1))
        
        num_inducing = self.inducing_variable.num_inducing

        sigma_sq = self.likelihood.variance_at(X)
        sigma = tf.sqrt(sigma_sq)

        kuf = Kuf(self.inducing_variable, self.kernel, X)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu) 

        Lkuf = tf.linalg.triangular_solve(L, kuf, lower=True)

        nu = (sigma_sq**-1)*Y*(W**2)-2*W*W_dy

        MKuf = tf.linalg.matmul(kuf,diag_W/sigma)

        A = kuu + tf.linalg.matmul(MKuf,MKuf,transpose_b=True)

        LA = tf.linalg.cholesky(A) 

        kufnu = tf.linalg.matmul(kuf,nu)

        tmp1 =tf.linalg.triangular_solve(LA, kufnu, lower=True)

        quad = tf.reduce_sum(tf.matmul(tmp1, tmp1, transpose_a=True))/2

        C1 = tf.matmul(Y,Y*(W**2)*self.likelihood.variance_at(X)**-1, transpose_a=True)/2
        C2 = - tf.matmul(Y,2*W*W_dy, transpose_a=True)
        C3 = - tf.matmul(W,W, transpose_a=True)
 
        C = tf.reduce_sum(C1+C2+C3)

        S = K - tf.matmul(Lkuf,Lkuf,transpose_a=True)
        LSL = tf.linalg.matmul(diag_W/sigma,tf.linalg.matmul(S,diag_W/sigma))

        trace = tf.reduce_sum(tf.linalg.trace(LSL)) 
        
        const = 0.5*to_default_float(num_inducing)*np.log(2*np.pi)

        det1 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LA)))
        det2 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        return quad - C - trace - det1 + det2 + const

    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        # could copy into posterior into a fused version

        X_data, Y_data = self.data

        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        
        W = self.diffusion_matrix.W(X_data, err)
        W_dy = self.diffusion_matrix.dy(X_data, err)

        diag_W = tf.linalg.diag(tf.squeeze(W, axis=-1))

        sigma_sq = self.likelihood.variance_at(X_data)
        sigma = tf.sqrt(sigma_sq)

        Yt = (sigma_sq**-1)*err*(W**2)-2*W*W_dy

        WKuf = tf.linalg.matmul(kuf,diag_W/sigma)

        L = tf.linalg.cholesky(kuu) 

        A = kuu + tf.linalg.matmul(WKuf,WKuf,transpose_b=True)

        LA = tf.linalg.cholesky(A) 

        tmp1 = tf.linalg.triangular_solve(LA, Kus, lower=True)
        tmp2 =tf.linalg.triangular_solve(LA, kuf, lower=True)

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