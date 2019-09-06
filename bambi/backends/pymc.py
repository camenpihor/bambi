"""Add PyMC3 as an available backend."""
from __future__ import absolute_import

import re
from collections import OrderedDict

import numpy as np
import pymc3 as pm
from pymc3.model import TransformedRV
import theano

from .base import BackEnd
from ..external.six import string_types
from ..priors import Prior
from ..results import MCMCResults, PyMC3ADVIResults


class PyMC3BackEnd(BackEnd):
    """PyMC3 model-fitting backend.

    Inherits from the BackEnd class.

    Attributes
    ----------
    links : dict
        Available link functions.
    dists : dict
        Available distributions.
    model : pm.Model
        A PyMC3 `Model` instance, which will be built up.
    mu : float
        mean
    spec : bambi.models.Model
        A bambi Model instance containing the abstract specification of the model to compile.
    trace : pymc3.backends.base.MultiTrace
        A `MultiTrace` object that contains the samples.
    advi_params : pm.variational.approximations.MeanFieldGroup
        A `MeanFieldGroup` object that contains the approximation to the posterior.
    """

    links = {
        "identity": lambda x: x,
        "logit": theano.tensor.nnet.sigmoid,
        "inverse": theano.tensor.inv,
        "inverse_squared": lambda x: theano.tensor.inv(theano.tensor.sqrt(x)),
        "log": theano.tensor.exp
    }

    dists = {"HalfFlat": pm.Bound(pm.Flat, lower=0)}

    def __init__(self):
        self.model = pm.Model()
        self.mu = None
        self.spec = None
        self.trace = None
        self.advi_params = None

    def reset(self):
        """Reset PyMC3 model and all tracked distributions and parameters."""
        self.model = pm.Model()
        self.mu = None
        self.spec = None
        self.trace = None
        self.advi_params = None

    def _expand_args(self, spec, key, value, label):
        if isinstance(value, Prior):
            label = f"{label}_{key}"
            return self._build_dist(spec, label, value.name, **value.args)
        return value

    def _build_dist(self, spec, label, dist, **kwargs):
        """Build and return the specified PyMC3 distribution.

        Build a distribution and its hyperparameters, if exist. Provide hyperparameters in **kwargs
        where the key is the desired label and the value is the PyMC3 Prior.

        Parameters
        ----------
        spec : bambi.models.Model
            A bambi Model instance containing the abstract specification of the model to compile.
        label : str
            Label used for the distribution.
        dist : str | pm.distributions
            PyMC3 distribution to build. If `str`, then attempt to find the PyMC3 distribution
            matching the string name.

        Returns
        -------
        pm.distributions
            The specified PyMC3 distribution.

        Raises
        ------
        ValueError
            Returned if `dist` is a string whose value is not a distribution in PyMC3.
        """
        # If `dist` is of type `str`, then find the matching PyMC3 distribution
        if isinstance(dist, string_types):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            elif dist in self.dists:
                dist = self.dists[dist]
            else:
                raise ValueError(
                    f"The Distribution class {dist} was not found in PyMC3 or the PyMC3BackEnd."
                )

        # Inspect all kwargs. If hyperparameter is found, build the hyper parameter distribution
        kwargs = {
            key: self._expand_args(
                spec=spec, key=key, value=value, label=label
            ) for (key, value) in kwargs.items()
        }

        # Handle non-centered parameterization for any hyperparameters
        if (
            spec.noncentered and
            "sd" in kwargs and
            "observed" not in kwargs and
            isinstance(kwargs["sd"], pm.model.TransformedRV)
        ):
            old_sd = kwargs["sd"]
            _offset = pm.Normal(label + "_offset", mu=0, sd=1, shape=kwargs["shape"])
            return pm.Deterministic(label, _offset * old_sd)

        return dist(label, **kwargs)

    def build(self, spec, reset=True):
        """Compile the PyMC3 model from an abstract model specification.

        Parameters
        ----------
        spec : bambi.models.Model
            A bambi Model instance containing the abstract specification of the model to compile.
        reset : bool, optional
            If True, reset the PyMC3BackEnd instance before compiling, by default True.
        """
        if reset:
            self.reset()

        with self.model:
            self.mu = 0.
            for model_term in spec.terms.values():
                data = model_term.data
                label = model_term.name
                dist_name = model_term.prior.name
                dist_args = model_term.prior.args
                n_cols = model_term.data.shape[1]

                coef_dist = self._build_dist(spec, label, dist_name,
                                        shape=n_cols, **dist_args)

                if model_term.random:
                    self.mu += coef_dist[model_term.group_index][:, None] * model_term.predictor
                else:
                    self.mu += pm.math.dot(data, coef_dist)[:, None]

            link_function = spec.family.link
            if isinstance(link_function, string_types):
                link_function = self.links[link_function]

            y_prior = spec.family.prior
            y_prior.args[spec.family.parent] = link_function(self.mu)
            y_prior.args["observed"] = spec.y.data

            self._build_dist(spec, spec.y.name, y_prior.name, **y_prior.args)
            self.spec = spec

    def _get_transformed_vars(self):
        # identify the variables that pymc3 back-transformed to original scale
        transformed_variables = [
            var.name
            for var in self.model.unobserved_RVs
            if isinstance(var, TransformedRV)
        ]

        # find the corresponding transformed variables
        transformed_variables = set(
            var.name
            for var in self.model.unobserved_RVs
            if any([x in var.name for x in transformed_variables])
        ) - set(transformed_variables)

        # add any "centered" random effects to the list
        for var_name in [var.name for var in self.model.unobserved_RVs]:
            if re.search(r'_offset$', var_name) is not None:
                transformed_variables.add(var_name)
        return transformed_variables

    def run(self, start=None, method="mcmc", init="auto", n_init=50_000, **sampler_kwargs):
        """Run the PyMC3 sampler.

        Parameters
        ----------
        start : dict, or array of dict, optional
            Starting parameter values to pass to the sampler; see pm.sample() documentation for
            details, by default None.
        method : str, optional
            The method to use for fitting the model. If "mcmc", the PyMC3 sampler will be used.
            Alternatively, "advi" designates that the model will be fit using automatic
            differentiation variational inference (ADVI) as implemented in PyMC3. By default "mcmc".
        init : str, optional
            Initialization method (see PyMC3 sampler documentation). Currently, their default is
            "jitter+adapt_diag". By default 'auto'.
        n_init : int, optional
            If init = 'advi' or 'nuts', then this designates the number of initialization
            iterations, by default 50,000.

        Returns
        -------
        PyMC3ModelResults
        """
        if method == "mcmc":
            num_samples = sampler_kwargs.pop("samples", 1_000)
            cores = sampler_kwargs.pop("chains", 1)
            with self.model:
                self.trace = pm.sample(draws=num_samples, start=start, init=init,
                                       n_init=n_init, cores=cores, **sampler_kwargs)
            return self._convert_to_results()

        elif method == "advi":
            with self.model:
                self.advi_params = pm.variational.ADVI(start=start, **sampler_kwargs)
            return PyMC3ADVIResults(self.spec, self.advi_params)

        raise ValueError(f"Invalid method ({method}) must be either 'advi' or 'mcmc'")

    def _get_levels(self, key, value):
        if len(value):
            # fixed effects
            if not self.spec.terms[re.sub(r"_offset$", "", key)].random:
                return self.spec.terms[key].levels

            # random effects
            else:
                re1 = re.match(r"(.+)(?=_offset)(_offset)", key)
                # handle "centered" terms
                if re1 is None:
                    return [
                        key.split("|")[0] + "|" + x
                        for x in self.spec.terms[key].levels
                    ]
                # handle "non-centered" terms
                else:
                    return [
                        "{}|{}_offset{}".format(
                            key.split("|")[0],
                            *re.match(r"^(.+)(\[.+\])$", x).groups()
                        )
                        for x in self.spec.terms[re1.group(1)].levels
                    ]
        return [key]

    def _convert_to_results(self):
        """Convert PyMC3 results to Bambi-readable results."""
        straces = self.trace._straces  # pylint: disable=protected-access

        pymc3_results = np.array(
            [
                np.array([np.atleast_2d(sample.T).T[:, chain_idx]
                for sample in straces[variable_idx].samples.values()
                for chain_idx in range(np.atleast_2d(sample.T).T.shape[1])])
                for variable_idx in range(len(straces))
            ]
        )
        pymc3_results = np.swapaxes(np.swapaxes(pymc3_results, 0, 1), 0, 2)

        # arange var_shapes dictionary in same order as samples dictionary
        shapes = OrderedDict()
        for key in straces[0].samples:
            shapes[key] = straces[0].var_shapes[key]

        # grab info necessary for making sample array pretty
        levels = sum(
            iterable=[self._get_levels(key=key, value=value) for key, value in shapes.items()],
            start=[]
        )

        return MCMCResults(
            model=self.spec,
            data=pymc3_results,
            names=list(shapes.keys()),
            dims=list(shapes.values()),
            levels=levels,
            transformed_vars=self._get_transformed_vars()
        )
