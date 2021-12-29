import numpy as np


class Population:
    def __init__(self, samples):
        self.samples = samples

    def fit_samples(self):
        for sample in self.samples:
            sample.fit_model()

    @property
    def best_sample(self):
        return sorted(self.sort())[0]

    @property
    def fitness(self):
        _population_fitness = []
        for sample in self:
            _population_fitness.append(sample.fitness)

        return _population_fitness

    @property
    def mutate_rates(self):
        fitness = np.array(self.fitness)
        rates = (fitness - min(fitness)) / (max(fitness) - min(fitness))

        for sample, mutate_rate in zip(self.samples, rates):
            sample._mutate_rate = mutate_rate

        return rates

    def fitness_statistics(self):
        """Return mean and std of population fitness"""
        population_fitness = self.fitness

        return (np.mean(population_fitness), np.std(population_fitness))

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.samples[key]
        elif isinstance(key, slice):
            return Population(self.samples[key])
        elif isinstance(key, Slice):
            raise Exception("error")

    def append(self, sample):
        if sample is None:
            raise Exception("sample is None")

        if isinstance(sample, Population):
            for samp in sample:
                self.samples.append(samp)
        else:
            self.samples.append(sample)

    def sort(self):
        self.samples = sorted(self.samples)

        return self
