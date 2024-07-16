import sys
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(SCRIPT_DIR)

import numpy as np
import gpflow

import rcgp.bo as bayesian_optimisation
import matplotlib.pyplot as plt
from tueplots import bundles
import time
import traceback
import gc
import tensorflow as tf
import psutil
import faulthandler
faulthandler.disable()

sys.setrecursionlimit(int(1e5))



CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


class Camel:
    def __init__(self, outliers=0.1, BASE_SEED=1234) -> None:
        self.max = 1.0316
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.dims = 2
        self.outliers = outliers
        self.name = 'Camel'
        self.seed = BASE_SEED
        self.rng = np.random.default_rng(BASE_SEED)

    def __call__(self, x):
        x1 = 4*x[:, 0]-2
        x2 = 2*x[:, 1]-1
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        y = term1 + term2 + term3
        if np.random.random() < self.outliers:
            epsilon = self.rng.uniform(1, 2)
            y += epsilon
            self.seed += 1
            self.rng = np.random.default_rng(self.seed)
        return -(y.reshape(-1, 1))


class Branin:
    def __init__(self, outliers=0.1, BASE_SEED=1234) -> None:
        self.max = 1.04739
        # self.bounds = np.array([[-5.0, 10.0], [1.0, 15.0]])
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.dims = 2
        self.a = 1
        self.b = 5.1/(4*np.pi**2)
        self.c = 5/np.pi
        self.r = 6
        self.s = 10
        self.t = 1/(8*np.pi)
        self.name = 'Branin'
        self.outliers = outliers
        self.seed = BASE_SEED
        self.rng = np.random.default_rng(BASE_SEED)

    def __call__(self, x):
        x1 = 15*x[:,0]-5
        x2 = 15*x[:,1]
        term1 = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2
        term2 = self.s*(1-self.t)*np.cos(x1)
        y = (term1 + term2 + self.s - 54.81)/51.95
        if self.rng.random() < self.outliers:
            epsilon = self.rng.uniform(1, 2)
            y += epsilon
            self.seed += 1
            self.rng = np.random.default_rng(self.seed)
        return -(y.reshape(-1, 1))


class Rosenbrock:
    def __init__(self, outliers=0.1, BASE_SEED=1234) -> None:
        self.max = 0
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.dims = 2
        self.name = 'Rosenbrock'
        self.outliers = outliers
        self.seed = BASE_SEED
        self.rng = np.random.default_rng(BASE_SEED)

    def __call__(self, x):
        x1 = 15*x[:,0]-5
        x2 = 15*x[:,1]-5
        term1 = 100 * (x2 - x1**2)**2
        term2 = (x1-1)**2
        y = (term1 + term2)/3.755e5
        if self.rng.random() < self.outliers:
            epsilon = self.rng.uniform(1, 2)
            y += epsilon
            self.seed += 1
            self.rng = np.random.default_rng(self.seed)
        return -(y.reshape(-1, 1))


class McCormick:
    def __init__(self, outliers=0.1, BASE_SEED=1234) -> None:
        self.max = 0.0956
        self.bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        self.dims = 2
        self.name = 'McCormick'
        self.outliers = outliers
        self.seed = BASE_SEED
        self.rng = np.random.default_rng(BASE_SEED)

    def __call__(self, x):
        x1 = 5.5*x[:, 0]-1.5
        x2 = 7*x[:, 1]-3
        term1 = np.sin(x1+x2)
        term2 = (x1-x2)**2
        term3 = -1.5*x1 + 2.5*x2 + 1
        y = (term1 + term2 + term3)/20
        if self.rng.random() < self.outliers:
            epsilon = self.rng.uniform(1, 2)
            y += epsilon
            self.seed += 1
            self.rng = np.random.default_rng(self.seed)
        return -(y.reshape(-1, 1))


def plot_bo(regrets, times, f, outliers, acc, functions, n):
    colors = ['#4daf4a', '#377eb8', '#ff7f00']
    gps = ['GP', 'RCGP', 't-GP']
    types = ['standard', 'robust', 'student-t']
    with plt.rc_context(bundles.aistats2023()):
        fig, ax = plt.subplots(ncols=len(functions), nrows=2, sharex=True, figsize=(1.75*len(functions), 2))
        for i, function in enumerate(functions):
            for j, type in enumerate(types):
                mean_time = np.cumsum(np.mean(np.array([time[function.name][type] for time in times]),axis=0))
                mean_regret = np.cumsum(np.mean(np.array([regret[function.name][type] for regret in regrets]),axis=0))

                std_time = np.std(np.array([time[function.name][type] for time in times]),axis=0)
                std_regrets = np.std(np.array([regret[function.name][type] for regret in regrets]),axis=0)
                ax[0, i].set_title(function.name)
                if type == 'robust':
                    ax[0, i].plot(mean_regret, label=gps[j], c=colors[j], ls='--', zorder=10, dashes=(5, 5))
                    ax[0, i].fill_between(
                            np.arange(len(mean_regret)),
                            mean_regret - 1.96 * std_regrets,
                            mean_regret + 1.96 * std_regrets,
                            facecolor=colors[j],
                            alpha=0.5)
                    ax[1, i].plot(mean_time, label=gps[j],  c=colors[j], ls='--', zorder=10, dashes=(5, 5))
                    ax[1, i].fill_between(
                            np.arange(len(mean_time)),
                            mean_time - 1.96 * std_time,
                            mean_time + 1.96 * std_time,
                            facecolor=colors[j],
                            alpha=0.5)
                else:
                    ax[0, i].plot(mean_regret, label=gps[j], c=colors[j])
                    ax[0, i].fill_between(
                            np.arange(len(mean_regret)),
                            mean_regret - 1.96 * std_regrets,
                            mean_regret + 1.96 * std_regrets,
                            facecolor=colors[j],
                            alpha=0.5)
                    ax[1, i].plot(mean_time, label=gps[j], c=colors[j])
                    ax[1, i].fill_between(
                            np.arange(len(mean_time)),
                            mean_time - 1.96 * std_time,
                            mean_time + 1.96 * std_time,
                            facecolor=colors[j],
                            alpha=0.5)
            ax[0, 0].set_ylabel('Cumulative \n regret')
            ax[1, 0].set_ylabel('Time [s]')
            ax[1, 0].set_xlabel('Iteration')
            ax[1, 1].set_xlabel('Iteration')
            ax[1, 2].set_xlabel('Iteration')
            ax[1, 3].set_xlabel('Iteration')
            ax[1, 0].legend()
            fig.savefig('figures/bo_outlier_{}_{}_{}_{}.pdf'.format(f, outliers, acc, n))


def run(f, outliers, acc, reps=5, n_iter=50, n_init=5, n_tries=10, BASE_SEED=1234, n=1):
    print('Running BO for {} iterations, {} Acc function and {}% Outliers.'.format(n_iter, acc, outliers*100))
    process = psutil.Process(os.getpid())
    types = ['robust', 'standard', 'student-t']
    regrets = []
    times = []

    C = {'Camel': 1,
         'Branin': 2,
         'Rosenbrock': 1,
         'McCormick': 1}

    acq_function = bayesian_optimisation.AcquisitionFunction(kind=acc)
    for rep in range(reps):
        print('REPETITION {}'.format(rep+1))
        random_seed = BASE_SEED + rep + n_iter * rep
        rng = np.random.default_rng(random_seed)
        if f == 'Camel':
            functions = [Camel(BASE_SEED=random_seed, outliers=outliers)]
        elif f == 'Branin':
            functions = [Branin(BASE_SEED=random_seed, outliers=outliers)]
        elif f == 'McCormick':
            functions = [McCormick(BASE_SEED=random_seed, outliers=outliers)]
        elif f == 'Rosenbrock':
            functions = [Rosenbrock(BASE_SEED=random_seed, outliers=outliers)]
        elif f == 'all':
            functions = [Camel(BASE_SEED=random_seed, outliers=outliers),
                         Branin(BASE_SEED=random_seed, outliers=outliers),
                         McCormick(BASE_SEED=random_seed, outliers=outliers),
                         Rosenbrock(BASE_SEED=random_seed, outliers=outliers)]
        regret_dict = {function.name: {type: [] for type in types} for function in functions}
        time_dict = {function.name: {type: [] for type in types} for function in functions}
        is_looping = True
        for function in functions:
            print('function', function.name)
            X_init = rng.uniform(function.bounds[:, 0], function.bounds[:, 1], size=(n_init, function.dims))
            Y_init = np.array([function(x.reshape(1, function.dims)) for x in X_init]).reshape(n_init, 1)
            for type in types:
                print('BO', type)
                function.seed = random_seed
                function.rng = np.random.default_rng(function.seed)
                kernel = gpflow.kernels.SquaredExponential()
                bo = bayesian_optimisation.BayesianOptimisation(X_init,
                                                                Y_init,
                                                                function,
                                                                acq_function,
                                                                kernel,
                                                                gp_kind=type,
                                                                bounds=function.bounds,
                                                                c=C[function.name],
                                                                df=10,
                                                                niter_gp=500,
                                                                lr=0.01)
                i = 0
                j = n_tries
                time_init = time.time()
                while i < n_iter:
                    bo.gp.kernel.variance.assign(1)
                    bo.gp.kernel.lengthscales.assign(1)
                    if i % 10 == 0:
                        print(f'Iteration {i}: rss {process.memory_info().rss >> 20} MB')
                    try:
                        time_before = time.time()
                        bo.optimisation(n_iter=1, n_restarts=2)
                        time_after = time.time()
                        regret_dict[function.name][type].append(np.abs(function.max-np.max(bo.Y_sample)))
                        time_dict[function.name][type].append(time_after - time_before)
                        i += 1
                    except Exception:
                        if j != 0:
                            try:
                                bo.X_next = rng.uniform(function.bounds[:, 0], function.bounds[:, 1], size=(1, function.dims))
                                bo.Y_next = function(bo.X_next)
                                bo.X_sample = np.vstack((bo.X_sample, bo.X_next))
                                bo.Y_sample = np.vstack((bo.Y_sample, bo.Y_next))
                                bo.update_gp()
                                regret_dict[function.name][type].append(np.abs(function.max-np.max(bo.Y_sample)))
                                if i > 0:
                                    time_dict[function.name][type].append(time_dict[function.name][type][-1])
                                else:
                                    time_dict[function.name][type].append(0)
                                j -= 1
                                i += 1
                                print('Try lefts: ', j)
                            except Exception:
                                is_looping = False
                                break
                        else:
                            is_looping = False
                            break
                print('Optimisation done in {}[s]'.format(time.time()-time_init))
                print('Final regrets: {}'.format(np.abs(function.max-np.max(bo.Y_sample))))
                del bo
                tf.keras.backend.clear_session()
                gc.collect()
                if not is_looping:
                    print('BREAK types loop')
                    break     
            if not is_looping:
                print('BREAK functions loop')
                break
        if is_looping:
            print('Add dicts')
            regrets.append(regret_dict)
            times.append(time_dict) 
            with open('results/bo_times_outliers_{}_{}_{}.pkl'.format(outliers, acc, f), 'wb') as file:
                pickle.dump(times, file)

            with open('results/bo_regrets_outliers_{}_{}_{}.pkl'.format(outliers, acc, f), 'wb') as file:
                pickle.dump(regrets, file)
            tf.keras.backend.clear_session()
            gc.collect()
    print('{} successful iterations'.format(len(times)))
    plot_bo(regrets=regrets,
            times=times,
            f=f,
            outliers=outliers,
            acc=acc,
            functions=functions,
            n=1)
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    run('all', 0.2, 'ucb', reps=15, n_iter=200, n_init=5, n_tries=20, BASE_SEED=1234)
    run('all', 0.2, 'poi', reps=20, n_iter=100, n_init=5, n_tries=20, BASE_SEED=1234)
    run('all', 0, 'ucb', reps=10, n_iter=100, n_init=5, n_tries=20, BASE_SEED=1234)
    run('all', 0, 'poi', reps=20, n_iter=100, n_init=5, n_tries=20, BASE_SEED=1234)