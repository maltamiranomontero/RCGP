from scipy.optimize import minimize
from scipy.stats import norm
import gpflow
from rcgp.rcgp import RCGPR
from rcgp.w import IMQ

import numpy as np


class AcquisitionFunction:
    def __init__(self,
                 kind,
                 xi=0.01,
                 kappa=2.576):
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
    def _expectedImprovement(X, Y_sample, gp, xi=0.01):

        mu, var = gp.predict_f(X, full_cov=False)
        sigma = np.sqrt(var)
        mu_sample_opt = np.max(Y_sample)

        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        if any(sigma == 0.0):
            ei[sigma == 0.0] = 0.0

        return ei

    @staticmethod
    def _upperConfidenceBounds(X, gp, kappa):

        mu, var = gp.predict_f(X, full_cov=False)
        sigma = np.sqrt(var)

        ucb = mu + kappa * sigma

        return ucb

    @staticmethod
    def _probabilityOfImprovement(X, Y_sample, gp, xi=0.01):

        mu, var = gp.predict_f(X, full_cov=False)
        sigma = np.sqrt(var)
        mu_sample_opt = np.max(Y_sample)

        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        poi = Z
        if any(sigma==0.0):
            poi[sigma == 0.0] = 0.0
        return poi


class BayesianOptimisation:
    def __init__(self,
                 f,
                 acquisition,
                 kernel,
                 gp_kind='stadard',
                 verbose=2,
                 bounds=None):
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

    def propose_location(self, X_sample, Y_sample, gp, n_restarts=25):

        dim = X_sample.shape[1]
        min_val = 1e10
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -self.acquisition(X.reshape(-1, dim), X_sample, Y_sample, gp).numpy().flatten()

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x.reshape(1, dim)

        return min_x

    def optimisation(self, n_iter, X_init, Y_init, n_restarts=25):
        # Initialize samples
        X_sample = X_init
        Y_sample = Y_init

        for i in range(n_iter):
            # Update Gaussian process with existing samples
            self.X_sample = X_sample
            self.Y_sample = Y_sample

            if self.gp_kind == 'standard':
                self.gp = gpflow.models.GPR(
                        (self.X_sample, self.Y_sample),
                        kernel=self.kernel,
                        noise_variance=1e-5)
                gpflow.set_trainable(self.gp.likelihood.variance, False)

            if self.gp_kind == 'robust':
                self.gp = RCGPR(
                        (self.X_sample, self.Y_sample),
                        kernel=self.kernel,
                        weighting_function=IMQ(C=2),
                        noise_variance=1e-5)
                gpflow.set_trainable(self.gp.likelihood.variance, False)

            if self.gp_kind == 'student-t':
                self.gp = gpflow.models.VGP(
                        (self.X_sample, self.Y_sample),
                        kernel=self.kernel,
                        likelihood=gpflow.likelihoods.StudentT(df=5))

            opt = gpflow.optimizers.Scipy()
            opt_options = dict()
            opt.minimize(self.gp.training_loss_closure(), self.gp.trainable_variables, options=opt_options)

            # Obtain next sampling point from the acquisition function
            self.X_next = self.propose_location(self.X_sample, self.Y_sample, self.gp, n_restarts)

            # Obtain next sample from the objective function
            self.Y_next = self.f(self.X_next)

            # Add sample to previous samples
            X_sample = np.vstack((self.X_sample, self.X_next))
            Y_sample = np.vstack((self.Y_sample, self.Y_next))

        return X_sample[np.argmax(Y_sample)], Y_sample[np.argmax(Y_sample)]
