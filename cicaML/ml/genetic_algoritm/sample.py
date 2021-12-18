import numpy as np
from abc import ABC, abstractclassmethod
from cicaML.ml.genetic_algoritm.population import Population


class Sample(ABC):
    def __init__(self, fitness=None):
        self._fitness = fitness
        self._mutate_rate = None

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __iter__(self):
        return iter(self.path)

    @property
    def fitness(self):
        if self._fitness == None:
            raise Exception("Fitness is none, please call the method fit")

        return self._fitness

    @property
    def mutate_rate(self):
        if self._mutate_rate == None:
            raise Exception(
                "Mutate rate is none, please call the method\
                            mutate_rates of population object that this sample\
                            belongs before calling this method"
            )

        return self._mutate_rate

    @abstractclassmethod
    def mutate(self, mutate_rate):
        pass

    @abstractclassmethod
    def fit_model(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def crossover(self, other):
        pass

    @abstractclassmethod
    def __copy__(self):
        pass

    @abstractclassmethod
    def save(self, filename=None):
        pass

    @classmethod
    def crossover_population(cls, population):
        population = population.sort()

        number_choose_samples = int(len(population) ** 0.5)
        choose_population = population[:number_choose_samples]
        random_samples = np.random.choice(
            population[number_choose_samples:], number_choose_samples - 1, False
        )

        new_population = Population([])
        new_population.append(choose_population)
        for s1 in choose_population:
            for s2 in random_samples:
                # print(s1.path)
                # print(s1.crossover(s2).path)
                new_population.append(s1.crossover(s2.mutate(0.4)))

        """ for s1, mutate_rate in zip(population, population.mutate_rates):
            new_population.append(s1.mutate(mutate_rate))"""

        return new_population
