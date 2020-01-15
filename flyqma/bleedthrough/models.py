import numpy as np
import statsmodels.api as sm
import warnings


class OLS:
    """
    Ordinary least squares model fit to X and Y data.

    Attributes:

        x, y (array like) - data

        model (sm.OLS) - fitted OLS model

    """

    def __init__(self, x, y, **fit_kw):
        """ Instantiate ordinary least squared model. """
        self.x = x
        self.y = y
        self.fit(**fit_kw)

    def fit(self, **kw):
        """ Fit model to X and Y data. """
        self.domain = np.linspace(0, self.x.max(), 10)
        x = sm.tools.add_constant(self.x.reshape(-1, 1))
        self.model = sm.OLS(self.y, x, hasconst=None).fit()

    def predict(self, x):
        """ Make model prediction. """
        xx = sm.tools.add_constant(x.reshape(-1, 1))
        return self.model.predict(xx)

    def detrend(self, x, y):
        """ Remove linear trend from X and Y data. """
        return y - self.predict(x)


class GLM(OLS):
    """
    Generalized linear model with gamma distributed residuals and an identity link function fit to X and Y data.

    Attributes:

        model (sm.genmod.GLM) - generalized linear model

        domain (np.ndarray[float]) - regularly spaced x-domain

    Inherited attributes:

        x, y (array like) - data

    """

    def fit(self, N=100000, maxiter=500, shift=0):
        """
        Fit Gamma GLM with identity link function.

        Args:

            N (int) - number of samples used

            maxiter (int) - maximum number of iterations for optimization

            shift (float) - offset used to keep values positive

        """

        self.domain = np.linspace(0, self.x.max(), 10)

        # downsample
        if N is not None:
            ind = np.random.randint(0, self.x.size, size=N)
        else:
            ind = np.arange(self.x.size)

        x, y = self.x[ind], self.y[ind]

        # construct variables
        xx = sm.tools.add_constant(x.reshape(-1, 1))
        yy = y + shift

        # define model
        family = sm.families.Gamma(link=sm.families.links.identity())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.genmod.GLM(yy, xx, family=family)

        # fit model
        start_params = [0.1+shift, 0.5]
        self.model = glm.fit(start_params=start_params, maxiter=maxiter)
        if not self.model.converged:
            raise Warning('GLM did not converge.')
