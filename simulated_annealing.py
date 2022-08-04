import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class SA:

    class _Individual:

        def __init__(self):

            self.position = {}
            self.solution_parsed = None
            self.cost = None


    def __init__(self,
                 cost_function,
                 type_number_of_variable,
                 max_iteration,
                 max_sub_iteration,
                 initial_temp,
                 temp_reduction_rate,
                 min_range_of_real_variables=0,
                 max_range_of_real_variables=1,
                 plot_solution=False,
                 number_of_population=1,
                 number_of_neighbors=1):

        self._cost_function = cost_function
        self._type_number_of_variables = type_number_of_variable
        self._max_iteration = max_iteration
        self._max_sub_iteration = max_sub_iteration
        self._initial_temp = initial_temp
        self._temp_reduction_rate = temp_reduction_rate
        self._min_range_of_real_variables = min_range_of_real_variables
        self._max_range_of_real_variables = max_range_of_real_variables
        self._plot_solution = plot_solution
        self._number_of_population = number_of_population
        self._number_of_neighbors = number_of_neighbors
        self._best_cost = []
        self._solution = None
        self._solution_neighbor = None

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.rand()
        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number < probs_cumsum)[0][0])


    def _initialize_evaluate(self):

        solution = self._Individual()

        def initialize_evaluate_permutation1D():

            solution.position["permutation1D"] = {}

            for j in range(len(self._type_number_of_variables["permutation1D"])):

                solution.position["permutation1D"][j] = \
                np.random.permutation(self._type_number_of_variables["permutation1D"][j]).\
                reshape(1, self._type_number_of_variables["permutation1D"][j])





        for type_of_variable in self._type_number_of_variables.keys():

            if type_of_variable == "permutation1D":

                initialize_evaluate_permutation1D()


        solution.solution_parsed, solution.cost = self._cost_function(solution.position)

        return solution

    def _create_neighbor(self):

        neighbor = self._Individual()

        def create_neighbor_permutation1D():

            neighbor.position["permutation1D"] = {}

            def create_neighbor_permutation1D_j(j):

                position = self._solution.position["permutation1D"][j]

                indices = [int(i) for i in np.random.choice(range(int(position.shape[1])), 2, replace=False)]

                min_index, max_index = min(indices), max(indices)

                def swap():

                    out = copy.copy(position)

                    out[0, [min_index, max_index]] = position[0, [max_index, min_index]]

                    return out

                def insertion():

                    method_index = self._roulette_wheel_selection(np.random.dirichlet([0.5, 0.5]))

                    if method_index == 0:

                        out = np.concatenate((position[:, :min_index + 1],
                                              position[:, max_index: max_index + 1],
                                              position[:, min_index + 1: max_index],
                                              position[:, max_index + 1:]), axis=1)

                        return out

                    elif method_index == 1:

                        out = np.concatenate((position[:, :min_index],
                                              position[:, min_index + 1: max_index + 1],
                                              position[:, min_index: min_index + 1],
                                              position[:, max_index + 1:]), axis=1)

                        return out

                def reversion():

                    out = np.concatenate((position[:, :min_index],
                                          np.flip(position[:, min_index: max_index + 1]),
                                          position[:, max_index + 1:]), axis=1)

                    return out

                method = self._roulette_wheel_selection(np.random.dirichlet([0.3, 0.3, 0.4]))

                if method == 0:

                    return swap()

                elif method == 1:

                    return insertion()

                elif method == 2:

                    return reversion()

            for j in range(len(self._type_number_of_variables["permutation1D"])):

                neighbor.position["permutation1D"][j] = create_neighbor_permutation1D_j(j)


        for type_of_variables in self._type_number_of_variables.keys():

            if type_of_variables == "permutation1D":

                create_neighbor_permutation1D()

        neighbor.solution_parsed, neighbor.cost = self._cost_function(neighbor.position)

        return neighbor


    def run(self):

        tic = time.time()

        assert callable(self._cost_function), "cost function should be callable."
        assert self._min_range_of_real_variables < self._max_range_of_real_variables, \
        "min range of real variables should be less than max range of real variables."

        self._solution = self._initialize_evaluate()

        for iter_main in range(self._max_iteration):

            for iter_sub in range(self._max_sub_iteration):

                self._solution_neighbor = self._create_neighbor()

                if self._solution_neighbor.cost < self._solution.cost:

                    self._solution = copy.deepcopy(self._solution_neighbor)

                else:

                    P = np.exp(-1 * ((self._solution_neighbor.cost - self._solution.cost) / self._solution.cost) /
                               self._initial_temp)

                    if np.random.rand() < P:

                        self._solution = copy.deepcopy(self._solution_neighbor)

            self._best_cost.append(self._solution.cost)
            self._initial_temp *= self._temp_reduction_rate


        toc = time.time()

        if self._plot_solution:

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))
            plt.plot(range(self._max_iteration), self._best_cost)
            plt.title("Objective Function Value Per Iteration Using Simulated Annealing", fontweight="bold")
            plt.xlabel("Number of Iteration")
            plt.ylabel("Best Cost")
            plt.savefig("./figures/cost_function_sa.png")

        return self._solution, self._solution_neighbor
