def no_improvement_for_x_generations(x):
    """
    Stop condition: no improvement for x generations
    """
    def stop_condition(genetic_algorithm):
        if len(genetic_algorithm.history["best_sample_score"]) < x:
            return False
        return (
            genetic_algorithm.history["best_sample_score"][-1] == genetic_algorithm.history["best_sample_score"][-x]
        )
    return stop_condition
