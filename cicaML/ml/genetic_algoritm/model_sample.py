import json
import numpy as np

from cicaML.ml.genetic_algoritm.sample import Sample
from cicaML.ml.genetic_algoritm.population import Population
from cicaML.utils.array import NumpyEncoder


def choice_sample_params(param_dict):

    if "value" in param_dict:
        return param_dict["value"]

    if "choice_callback" in param_dict:
        return param_dict["choice_callback"](param_dict)

    variations = param_dict["variations"]

    if "sample_size" in param_dict:
        sample_size = param_dict["sample_size"]
        if sample_size == 1:
            return np.random.choice(variations, size=1, replace=False)[0]
        else:
            if "default" in param_dict:
                default = param_dict["default"]
                params = np.random.choice(
                    variations, size=sample_size - len(default), replace=False
                )
                params = np.append((params, default))

                return np.unique(params).tolist()

            return np.random.choice(
                variations, size=sample_size, replace=False
            ).tolist()

    min_sample_size = param_dict.get("min_sample_size", 0)
    max_sample_size = param_dict.get("max_sample_size", len(variations))

    sample_size = np.random.randint(min_sample_size, max_sample_size)

    params = np.random.choice(variations, size=sample_size, replace=False)

    if "default" in param_dict:
        params = np.append(params, param_dict["default"])

    return np.unique(params).tolist()


def choose_probability(sample_1, sample_2):
    if sample_1.fitness is None:
        return 0.5
    if sample_2.fitness is None:
        return 0.5

    return sample_1.fitness / (sample_1.fitness + sample_2.fitness)


class ModelChoiceSample(Sample):
    def __init__(
        self,
        params,
        params_universe,
        fit_model_func,
        fitness=None,
    ):
        self.params = params
        self.params_universe = params_universe
        self.fit_model_func = fit_model_func
        self.summary = None
        self._fitness = fitness

    def mutate(self, mutate_rate):
        params = {}
        for param in self.params_universe:
            if np.random.rand() < mutate_rate:
                params[param] = choice_sample_params(self.params_universe[param])
            else:
                params[param] = self.params[param]

        return ModelChoiceSample(params, self.params_universe, self.fit_model_func)

    def fit_model(self, *args, **kwargs):
        if self._fitness is None:
            fitness, summary = self.fit_model_func(self)
            self._fitness = fitness
            self.summary = summary

    def crossover(self, other):
        params = {}
        probability_self = choose_probability(self, other)
        for param in self.params_universe:
            if np.random.rand() <= probability_self:
                params[param] = self.params[param]
            else:
                params[param] = other.params[param]

        return ModelChoiceSample(params, self.params_universe, self.fit_model_func)

    def save(self, filename):
        json.dump(self.summary, open(f"{filename}_summary.json", "w"), cls=NumpyEncoder)

    def __copy__(self):
        pass

    def __str__(self):
        return str(self.params)


def generate_init_population(
    population_size,
    params_universe,
    fit_model_func,
):
    population = Population([])
    for i in range(population_size):
        params = {}
        for param_name, param_dict in params_universe.items():
            params[param_name] = choice_sample_params(param_dict)

        sample = ModelChoiceSample(
            params=params,
            params_universe=params_universe,
            fitness=None,
            fit_model_func=fit_model_func,
        )

        population.append(sample)

    return population
