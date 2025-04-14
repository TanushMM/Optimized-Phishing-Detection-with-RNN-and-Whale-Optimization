import numpy as np

class WhaleOptimization:
    def __init__(self, opt_func, constraints, nsols=10, b=1.5, a=2.0, a_step=0.1, maximize=False):
        """
        Initialize the Whale Optimization Algorithm.

        :param opt_func: Fitness function to minimize or maximize
        :param constraints: Bounds for each parameter (list of tuples)
        :param nsols: Number of whales (solutions)
        :param b: Spiral constant
        :param a: Initial exploration factor
        :param a_step: Value by which 'a' is reduced every iteration
        :param maximize: Set True for maximization problems
        """
        self._opt_func = opt_func
        self._constraints = np.array(constraints)
        self._nsols = nsols
        self._b = b
        self._a = a
        self._a_step = a_step
        self._maximize = maximize
        self._best_solution = None
        self._best_fitness = float('-inf') if maximize else float('inf')

        self._sols = np.random.uniform(
            [low for (low, high) in self._constraints],
            [high for (low, high) in self._constraints],
            size=(self._nsols, len(self._constraints))
        )

    def _rank_solutions(self):
        """
        Evaluate and rank all solutions based on the fitness function.
        """
        fitness_values = []
        for sol in self._sols:
            try:
                fitness = self._opt_func(sol)
                fitness_values.append(fitness)
            except Exception as e:
                print(f"❌ Fitness evaluation failed: {e}")
                fitness_values.append(float('inf'))

        fitness_values = np.array(fitness_values)

        # 🔹 Update best solution
        if self._maximize:
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self._best_fitness:
                self._best_fitness = fitness_values[best_idx]
                self._best_solution = self._sols[best_idx]
        else:
            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < self._best_fitness:
                self._best_fitness = fitness_values[best_idx]
                self._best_solution = self._sols[best_idx]

        print(f"\n✅ Best solution so far: ({self._best_fitness}, {self._best_solution})")
        return fitness_values

    def optimize(self, iterations=2):
        """
        Main loop for WOA optimization.

        :param iterations: Number of iterations
        """
        for iter_num in range(iterations):
            print(f"\n🔄 Iteration {iter_num + 1}/{iterations}")
            fitness_values = self._rank_solutions()

            a = self._a - iter_num * self._a_step 

            for i in range(self._nsols):
                A = 2 * a * np.random.rand() - a
                C = 2 * np.random.rand()

                whale = self._sols[i]
                best = self._best_solution

                if best is None or whale.shape != best.shape:
                    continue 

                if np.random.rand() < 0.5:
                    D = np.abs(C * best - whale)
                    new_position = best - A * D
                else:
                    D = np.abs(best - whale)
                    l = np.random.uniform(-1, 1)
                    new_position = D * np.exp(self._b * l) * np.cos(2 * np.pi * l) + best

                self._sols[i] = np.clip(new_position, self._constraints[:, 0], self._constraints[:, 1])

    def get_best_solution(self):
        """
        Returns the best fitness and corresponding solution.
        """
        return self._best_fitness, self._best_solution
