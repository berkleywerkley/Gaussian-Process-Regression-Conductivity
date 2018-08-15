"""
Returns values of ep24 and ss_ratio to explore next

reference: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
"""

import numpy as np
import sklearn.gaussian_process as gp 

from scipy.stats import norm
from scipy.optimize import minimize

import get_data

def expected_improvement(x, gaussian_process, conductivity, n_params=3):
    """
    Expected improvement int conductivity

    x: array_like, shape = [n_samples, n_hyperparams]
        The point for which the expected improvement in conductivity needs to be computed
    gaussian_process:
        Gaussian process trained on previously evaluated values of eps24 and ss_radius
    greater_is_better: Boolean.
        Boolean flag that indicates whether the loss function is to be maximised or minimised.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)
    print('mu: ',mu)
    print('sigma ', sigma)

    loss_optimum = np.max(conductivity)

    #In case that sigma equals zero
    with np.errstate(divide='ignore'):
        Z = (mu - loss_optimum) / sigma
        expected_improvement = (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement

def sample_next_hyperparameter(acquisition_func, gaussian_process, conductivity, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise. In this case it is the function 'expected improvement'
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, conductivity, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

def bayesian_optimisation(data, bounds, n_iters=1, alpha=1e-5, epsilon=1e-7):

    """ bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        data: dictionary
            key = tuple of eps24 and ratio
            value = conductivity
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        n_iters: integer.
            Number of iterations to run the search algorithm.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    x_list = []
    y_list = []

    for params in data:
        x_list.append(params)
        y_list.append(data[params])
    print('loaded data')
    xp = np.array(x_list)
    yp = np.array(y_list)

    #create the GP
    kernel = gp.kernels.Matern() #Matern Kernel assumes that function is at least once differentiable. The main assumption
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    model.fit(xp, yp)

    next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=False, bounds=bounds, n_restarts=100)

    # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
    if np.any(np.abs(next_sample - xp) <= epsilon):
        next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])    

    return next_sample


def average(array):
    return sum(array)/len(array)

if __name__ == '__main__':
    data = get_data.main()

    bounds = np.array([[10,24], [0.33, 0.65], [2,10]])

    eps24 = []
    radius =[]
    eps34 = []

    for i in range(10):
        print('iteration %d' % ((i+1)))
        x, y, z = bayesian_optimisation(data, bounds)
        eps24.append(x)
        radius.append(y)
        eps34.append(z)
    
    print((average(eps24), average(radius), average(eps34)))