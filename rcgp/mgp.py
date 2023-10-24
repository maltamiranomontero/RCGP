import numpy as np
import numpy.matlib
from scipy.special import psi
from scipy.spatial import distance_matrix

# Implementation of 'Robust and Scalable Gaussian Process Regression and Its Applications' based on the
# original implementation github.com/YifanLu2000/Robust-Scalable-GPR


class MGP:
    def __init__(self, data,):
        self.data = data
        self.beta = 1
        self.lambda_ = 1
        self.outlierA = None
        self.minP = 1e-8
        self.Ba = 10
        self.Bb = 10

    def fit(self, niter=500, verbose=False):

        X, Y = self.data
        # get data num
        N = X.shape[0]
        # initialization
        f = np.zeros((N, 1))

        sigma2 = sum((Y - f) ** 2) / N

        Ba = self.Ba
        Bb = self.Bb

        Gamma = np.exp((Ba) - psi(Ba + Bb))
        minusGamma = np.exp(psi(Bb + N) - psi(Ba + Bb + N))
        variancef = np.zeros((N, 1))

        momentum = 0.9

        gradLambda = 0
        gradBeta = 0

        outlierA = np.max(Y)-np.min(Y)

        minP = self.minP

        beta = self.beta
        lambda_ = self.lambda_
        iter = 1

        K, dist_2 = self.construct_kernel(X, X, lambda_, beta)
        # start main loop of inference
        while iter < niter:
            if verbose:
                print(iter)
            stepSize = 1 / (iter + 100.0)
            # update q_3(Z)
            P = self.get_P(Y, f, sigma2, Gamma, minusGamma, variancef, outlierA)
            P[P < minP] = minP
            Sp = sum(P)
            # update q_2(\gamma)
            Gamma = np.exp(psi(Ba + Sp) - psi(Ba + Bb + N))
            minusGamma = np.exp(psi(Bb + N - Sp) - psi(Ba + Bb + N))
            # update q_1(f,f_m)
            commonTerm = np.linalg.inv(np.eye(N) + K @ np.diag(P[:, 0]) / sigma2 + np.eye(N)*1e-6)
            f = K @ commonTerm @ (Y * P/ sigma2) 

            variancef = (lambda_ + np.diag(K @ commonTerm) - np.diag(K))
            variancef[variancef <= 0] = 0
            variancef = variancef.reshape(-1, 1)
            # update q_4(sigma^2, lambda, beta, Xm)
            sigma2 = sum(((Y - f) ** 2.0 ) * P + P*variancef) / Sp


            K_y = sigma2 * np.diag(P[:, 0]**-1) + K
            K_y_inv = np.linalg.inv(K_y+np.eye(N)*1e-6)
            C = K_y_inv@Y
            # calculate the gradient for lambda
            dKy_dlambda = K / lambda_
            gradLambda = 0.5 * momentum * (np.trace(C @ np.transpose(C) @ dKy_dlambda) - np.trace(K_y_inv@dKy_dlambda) - Sp / sigma2 + np.trace(np.diag(P[:,0]) @ dKy_dlambda) / sigma2) + (1 - momentum) * gradLambda
            gradLambda[gradLambda > 1] = 1
            gradLambda[gradLambda < -1] = -1
            #     lambda = lambda + stepSize * gradLambda;
            lambda_ = max(lambda_ + stepSize * gradLambda, 0.1)
            # calculate the gradient for lambda
            dKy_dbeta = - np.multiply(K,dist_2)
            gradBeta = 0.5 * momentum * (np.trace(C @ np.transpose(C) @ dKy_dbeta) - np.trace(K_y_inv@dKy_dbeta) + np.trace(np.diag(P[:,0]) @ dKy_dbeta) / (sigma2)) + (1 - momentum) * gradBeta
            gradBeta[gradBeta > 1] = 1
            gradBeta[gradBeta < -1] = -1
            #     beta = beta + stepSize * gradBeta;
            beta = max(beta + stepSize * gradBeta, 0.1)
            # update kernel
            K, dist_2 = self.construct_kernel(X, X, lambda_, beta)
            iter = iter + 1


        # output hyperparamter of GP
        SigmaInv = K + K @ np.diag(P[:,0]) @ K / sigma2
        mu = K@np.linalg.inv(SigmaInv+np.eye(N)*1e-6) @ K @ (P * Y / sigma2)

        self.lambda_ = lambda_
        self.beta = beta
        self.sigma2 = sigma2
        self.P = P
        self.mu = mu
        self.SigmaInv = SigmaInv
        self.f = f
        self.K = K

    def predict_f(self, X):
        U,__ = self.construct_kernel(X, self.data[0], self.lambda_, self.beta)
        N = self.K.shape[0]
        K_inv = np.linalg.inv(self.K+np.eye(N)*1e-6)
        mean = U@K_inv @ self.mu
        variance = (self.lambda_ - np.diag(U @ K_inv @ np.transpose(U)) + np.diag(U @ np.linalg.inv(self.SigmaInv+np.eye(N)*1e-6) @ np.transpose(U))).reshape(-1, 1)

        return mean, variance

    @staticmethod
    def get_P(Y, f, sigma2, Gamma, minusGamma, variancef, outlierA):
        inlier_term = np.multiply(Gamma * np.exp(- (Y - f) ** 2 / (2 * sigma2)), np.exp(- variancef / (2 * sigma2))) / np.sqrt(2 * np.pi * sigma2)
        P = inlier_term / (minusGamma / outlierA + inlier_term)
        return P

    @staticmethod
    def construct_kernel(Xn, Xm, lambda_, beta):
        dist_nm2 = distance_matrix(Xn, Xm, p=2)** 2
        Knm = lambda_ * np.exp(- beta * dist_nm2)
        return Knm, dist_nm2
