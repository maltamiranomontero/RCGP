from scipy.optimize import minimize
from scipy.stats import norm
import gpflow
from rcgp.rcgp import RCGPR
from rcgp.w import IMQ
import tensorflow as tf

import numpy as np
import time

class AcquisitionFunction:
    def __init__(self,
                 kind,
                 xi=1e-4,
                 kappa=10):
        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind
        self.xi = xi
        self.kappa = kappa

    def __call__(self, X, X_sample, Y_sample, gp):
        if self.kind == 'ei':
            return self._expectedImprovement(X, Y_sample, gp, self.xi)
        if self.kind == 'ucb':
            return self._upperConfidenceBounds(X, gp, self.kappa)
        if self.kind == 'poi':
            return self._probabilityOfImprovement(X, Y_sample, gp, self.xi)

    @staticmethod
    def _expectedImprovement(X, Y_sample, gp, xi=1e-4):

        mu, var = gp.posterior().predict_f(X, full_cov=False)
        sigma = tf.sqrt(var)
        mu_sample_opt = tf.reduce_max(Y_sample)

        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        if any(sigma == 0.0):
            ei[sigma == 0.0] = 0.0

        return ei

    @staticmethod
    def _upperConfidenceBounds(X, gp, kappa=10):

        mu, var = gp.posterior().predict_f(X, full_cov=False)
        sigma = tf.sqrt(var)

        ucb = mu + kappa * sigma

        return ucb

    @staticmethod
    def _probabilityOfImprovement(X, Y_sample, gp, xi=0.1):

        mu, var = gp.posterior().predict_f(X, full_cov=False)
        sigma = tf.sqrt(var)
        mu_sample_opt = tf.reduce_max(Y_sample)

        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        poi = Z
        return poi


class BayesianOptimisation:
    def __init__(self,
                 X_sample,
                 Y_sample,
                 f,
                 acquisition,
                 kernel,
                 gp_kind='stadard',
                 verbose=2,
                 bounds=None,
                 c=2,
                 df=10,
                 niter_gp=1000,
                 lr=0.01):
        self.X_sample = X_sample
        self.Y_sample = Y_sample
        self.f = f
        self.acquisition = acquisition
        self.kernel = kernel
        if gp_kind not in ['standard', 'robust', 'student-t']:
            err = "please choose one of standard, robust, student-t"
            raise NotImplementedError(err)
        else:
            self.gp_kind = gp_kind
        self.verbose = verbose
        self.bounds = bounds
        self.c = c
        self.df = df
        self.niter_gp = niter_gp
        self.lr = lr

        if self.gp_kind == 'standard':
            self.gp = gpflow.models.GPR(
                    (self.X_sample, self.Y_sample),
                    kernel=self.kernel,
                    noise_variance=1e-4)
            # gpflow.set_trainable(self.gp.likelihood.variance, False)

        if self.gp_kind == 'robust':
            self.gp = RCGPR(
                    (self.X_sample, self.Y_sample),
                    kernel=self.kernel,
                    weighting_function=IMQ(C=self.c),
                    noise_variance=1e-4)
            # gpflow.set_trainable(self.gp.likelihood.variance, False)
            gpflow.set_trainable(self.gp.weighting_function.C, False)

        if self.gp_kind == 'student-t':
            self.gp = gpflow.models.vgp.VGP(
                    (self.X_sample, self.Y_sample),
                    kernel=self.kernel,
                    likelihood=gpflow.likelihoods.StudentT(scale=1e-4, df=self.df))

    def propose_location(self, X_sample, Y_sample, gp, n_restarts=25):

        dim = X_sample.shape[1]
        min_val = 1e10
        min_x = None

        def min_obj(X):
            X = tf.reshape(X, (1, dim))
            return -self.acquisition(X, X_sample, Y_sample, gp)

        @tf.function
        def val_and_grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = min_obj(x)
            grad = tape.gradient(loss, x)
            return loss, grad

        def func(x):
            return [vv.numpy().astype(np.float64) for vv in val_and_grad(tf.constant(x, dtype=tf.float64))]

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, dim)):
            opt = {'maxfun': 100, 'maxiter': 20}
            res = minimize(func, x0=x0, bounds=self.bounds, method='L-BFGS-B',
                           options=opt, jac=True)
            #print('N propose: {} . N func eval: {}'.format(res.nit,res.nfev))
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x.reshape(1, dim)
        return min_x

    @staticmethod
    def run_adam(model, iterations, lr):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        # Create an Adam Optimizer action
        optimizer = tf.optimizers.legacy.Adam(learning_rate=lr)

        @tf.function
        def optimization_step():
            optimizer.minimize(model.training_loss_closure(), model.trainable_variables)

        for step in range(iterations):
            optimization_step()

    def optimisation(self, n_iter, n_restarts=25):
        for _ in range(n_iter):
            #self.run_adam(self.gp, self.niter_gp, self.lr)
            # a = time.time()
            self.update_gp()
            opt = gpflow.optimizers.Scipy()
            opt.minimize(self.gp.training_loss_closure(), self.gp.trainable_variables)  
            #print('N optimisation: ',res.nit)     
            # print('Learning Phase: ', time.time()-a)
            # Obtain next sampling point from the acquisition function
            # a = time.time()
            self.X_next = self.propose_location(self.X_sample, self.Y_sample,
                                                self.gp, n_restarts)
            # print('Propose Phase: ', time.time()-a)

            # Obtain next sample from the objective function
            self.Y_next = self.f(self.X_next)

            # Add sample to previous samples
            self.X_sample = np.vstack((self.X_sample, self.X_next))
            self.Y_sample = np.vstack((self.Y_sample, self.Y_next))
        return

    def update_gp(self):
        self.gp.data = gpflow.models.util.data_input_to_tensor((self.X_sample, self.Y_sample))
        if self.gp_kind == 'student-t':
            static_num_data = self.gp.data[0].shape[0]
            self.gp.num_data.assign(int(static_num_data))

            dynamic_num_data = tf.convert_to_tensor(self.gp.num_data)
            num_latent_gps = self.gp.calc_num_latent_gps_from_data(self.gp.data, self.gp.kernel, self.gp.likelihood)
            q_sqrt_unconstrained_shape = (num_latent_gps, gpflow.utilities.triangular_size(static_num_data))
            q_mu = gpflow.base.Parameter(
                tf.zeros((dynamic_num_data, num_latent_gps)),
                shape=(static_num_data, num_latent_gps),
            )
            q_sqrt = tf.eye(dynamic_num_data, batch_shape=[self.gp.num_latent_gps])
            q_sqrt = gpflow.base.Parameter(
                q_sqrt,
                transform=gpflow.utilities.triangular(),
                unconstrained_shape=q_sqrt_unconstrained_shape,
                constrained_shape=(num_latent_gps, static_num_data, static_num_data),
            )
            self.gp.q_mu = q_mu
            self.gp.q_sqrt = q_sqrt
        return
