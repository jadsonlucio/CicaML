import matplotlib.pyplot as plt


class Genetic:
    def __init__(self, population):
        self.history = {
            "mean_samples_score": [],
            "std_samples_score": [],
            "best_sample_score": [],
        }

        self.sample_class = population[0].__class__
        self.population = population

    def step(self):
        pass

    def run(
        self,
        generation_callback=None,
        max_generations=2000,
        verbose=False,
        plot_results=False,
        stop_condition=None
    ):
        cont = 0
        while cont < max_generations:
            if cont == 0:
                self.population.fit_samples()

            self.population = self.sample_class.crossover_population(self.population)
            self.population.fit_samples()
            (
                mean_fitness_population,
                std_fitness_population,
            ) = self.population.fitness_statistics()

            self.history["mean_samples_score"].append(mean_fitness_population)
            self.history["std_samples_score"].append(std_fitness_population)
            self.history["best_sample_score"].append(self.population[0].fitness)

            cont += 1

            if verbose:
                print(f"generation:{cont}")

            if generation_callback:
                generation_callback(cont, self.history, self.population)

            if stop_condition and stop_condition(self):
                break

        if plot_results:
            self.plot_history(["best_sample_score", "mean_samples_score"])

    def plot_history(self, key):
        if isinstance(key, list):
            for k in key:
                plt.plot(self.history[k], label=k)
        else:
            plt.plot(self.history[key], label=key)

        plt.legend()
        plt.show()
