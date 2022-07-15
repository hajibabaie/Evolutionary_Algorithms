import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class GA:

    class _Individual:

        def __init__(self):

            self.position = {}
            self.solution_parsed = {}
            self.cost = None

    def __init__(self,
                 cost_function,
                 type_number_of_variables,
                 min_range_of_real_variables=0,
                 max_range_of_real_variables=1,
                 min_range_of_integer_variables=0,
                 max_range_of_integer_variables=1,
                 max_iteration=100,
                 number_of_population=20,
                 parents_selection_method="roulette_wheel_selection",
                 selection_pressure_for_roulette_wheel=1,
                 tournament_size=2,
                 crossover_method="single_point",
                 crossover_percentage=0.5,
                 mutation_percentage=0.5,
                 mutation_rate=0.1,
                 gamma_for_crossover=0.1,
                 gamma_for_mutation=0.1,
                 plot_cost_function=False,
                 y_axis="linear"):


        self._cost_function = cost_function
        self._type_number_of_variables = type_number_of_variables
        self._min_range_of_real_variables = min_range_of_real_variables
        self._max_range_of_real_variables = max_range_of_real_variables
        self._min_range_of_integer_variables = min_range_of_integer_variables
        self._max_range_of_integer_variables = max_range_of_integer_variables
        self._min_range_of_binary_variables = 0
        self._max_range_of_binary_variables = 1
        self._max_iteration = max_iteration
        self._number_of_population = number_of_population
        self._parents_selection_method = parents_selection_method
        self._selection_pressure = selection_pressure_for_roulette_wheel
        self._tournament_size = tournament_size
        self._crossover_method = crossover_method
        self._population_main = None
        self._population_main_probs = None
        self._crossover_percentage = crossover_percentage
        self._number_of_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_of_population / 2))
        self._population_crossover = None
        self._mutation_percentage = mutation_percentage
        self._number_of_mutation = int(np.ceil(self._mutation_percentage * self._number_of_population))
        self._population_mutation = None
        self._mutation_rate = mutation_rate
        self._gamma_for_crossover = gamma_for_crossover
        self._gamma_for_mutation = gamma_for_mutation
        self._plot_cost_function = plot_cost_function
        self._worst_cost = None
        self._best_cost = []
        self._yaxis = y_axis

    def _initialize_population(self):

        population_main = [self._Individual() for _ in range(self._number_of_population)]

        def initialize_population_binary1D(population):

            for i in range(len(population)):

                population[i].position["binary1D"] = {}

                for j in range(len(self._type_number_of_variables["binary1D"])):

                    population[i].position["binary1D"][j] = \
                    np.random.randint(0, 2, (1, self._type_number_of_variables["binary1D"][j]))

            return population

        def initialize_population_real1D(population):

            for i in range(len(population)):

                population[i].position["real1D"] = {}

                for j in range(len(self._type_number_of_variables["real1D"])):

                    population[i].position["real1D"][j] = \
                    np.random.uniform(self._min_range_of_real_variables,
                                      self._max_range_of_real_variables,
                                      (1, self._type_number_of_variables["real1D"][j]))
            return population

        for type_of_variable in self._type_number_of_variables.keys():

            if type_of_variable == "binary1D":

                population_main = initialize_population_binary1D(population_main)

            if type_of_variable == "real1D":

                population_main = initialize_population_real1D(population_main)

        return population_main

    def _evaluate_population(self, population):

        for i in range(len(population)):

            population[i].solution_parsed, \
            population[i].cost = self._cost_function(population[i].position)

        return population

    @staticmethod
    def _sort(population):

        population = [population[i] for i in np.argsort([pop.cost for pop in population])]

        return population

    @staticmethod
    def _roulette_wheel_selection(probabilities):

        random_number = np.random.rand()

        probabilities_cumulative_sum = np.cumsum(probabilities)

        return int(np.argwhere(random_number <= probabilities_cumulative_sum)[0][0])

    def _tournament_selection(self):

        selected_population = np.random.choice(range(self._number_of_population), self._tournament_size, replace=False)

        selected_population_cost_argmin = np.argmin([self._population_main[i].cost for i in selected_population])

        return int(selected_population[selected_population_cost_argmin])

    def _population_probabilities_calculations(self):

        population_cost = [pop.cost for pop in self._population_main]

        worst_cost = max(population_cost)

        worst_cost = max(worst_cost, self._worst_cost)

        population_cost = np.divide(population_cost, worst_cost)

        probabilities = np.exp(-1 * self._selection_pressure * population_cost)

        probabilities_sum = np.sum(probabilities)

        probabilities = np.divide(probabilities, probabilities_sum)

        return probabilities

    def _apply_crossover(self):

        population_crossover = [self._Individual() for _ in range(self._number_of_crossover)]

        def apply_crossover_binary1D(population):

            for i in range(0, len(population), 2):

                population[i].position["binary1D"] = {}
                population[i + 1].position["binary1D"] = {}

                if self._parents_selection_method == "roulette_wheel_selection":

                    first_parent_index = self._roulette_wheel_selection(self._population_main_probs)
                    second_parent_index = self._roulette_wheel_selection(self._population_main_probs)
                    while first_parent_index == second_parent_index:
                        second_parent_index = self._roulette_wheel_selection(self._population_main_probs)

                else:

                    first_parent_index = self._tournament_selection()
                    second_parent_index = self._tournament_selection()
                    while first_parent_index == second_parent_index:
                        second_parent_index = self._tournament_selection()



                for j in range(len(self._type_number_of_variables["binary1D"])):

                    parent_first = self._population_main[first_parent_index].position["binary1D"][j]
                    parent_second = self._population_main[second_parent_index].position["binary1D"][j]

                    if self._crossover_method == "single_point":

                        cut_point = int(np.random.choice(range(1, self._type_number_of_variables["binary1D"][j])))

                        population[i].position["binary1D"][j] = \
                        np.concatenate((parent_first[:, :cut_point],
                                        parent_second[:, cut_point:]), axis=1)

                        population[i + 1].position["binary1D"][j] = \
                        np.concatenate((parent_second[:, :cut_point],
                                        parent_first[:, cut_point:]), axis=1)

                    elif self._crossover_method == "double_point":

                        cut_points = \
                        [int(i) for i in np.random.choice(range(1, self._type_number_of_variables["binary1D"][j]), 2,
                                                          replace=False)]

                        cut_point_first, cut_point_second = min(cut_points), max(cut_points)

                        population[i].position["binary1D"][j] = \
                        np.concatenate((parent_first[:, :cut_point_first],
                                        parent_second[:, cut_point_first:cut_point_second],
                                        parent_first[:, cut_point_second:]), axis=1)

                        population[i + 1].position["binary1D"][j] = \
                        np.concatenate((parent_second[:, :cut_point_first],
                                        parent_first[:, cut_point_first:cut_point_second],
                                        parent_second[:, cut_point_second:]), axis=1)

                    else:

                        alpha = np.random.randint(0, 2, parent_first.shape)

                        population[i].position["binary1D"][j] = \
                        np.multiply(alpha, parent_first) + np.multiply(1 - alpha, parent_second)

                        population[i + 1].position["binary1D"][j] = \
                        np.multiply(alpha, parent_second) + np.multiply(1 - alpha, parent_first)

            return population

        def apply_crossover_real1D(population):

            for i in range(0, len(population), 2):

                population[i].position["real1D"] = {}
                population[i + 1].position["real1D"] = {}

                if self._parents_selection_method == "roulette_wheel_selection":

                    first_parent_index = self._roulette_wheel_selection(self._population_main_probs)
                    second_parent_index = self._roulette_wheel_selection(self._population_main_probs)
                    while first_parent_index == second_parent_index:
                        second_parent_index = self._roulette_wheel_selection(self._population_main_probs)

                else:

                    first_parent_index = self._tournament_selection()
                    second_parent_index = self._tournament_selection()
                    while first_parent_index == second_parent_index:
                        second_parent_index = self._tournament_selection()

                for j in range(len(self._type_number_of_variables["real1D"])):

                    parent_first = self._population_main[first_parent_index].position["real1D"][j]
                    parent_second = self._population_main[second_parent_index].position["real1D"][j]

                    alpha = np.random.uniform(-self._gamma_for_crossover, 1 + self._gamma_for_crossover,
                                              parent_first.shape)

                    population[i].position["real1D"][j] = \
                    np.multiply(alpha, parent_first) + np.multiply(1 - alpha, parent_second)

                    population[i + 1].position["real1D"][j] = \
                    np.multiply(alpha, parent_second) + np.multiply(1 - alpha, parent_first)

                    population[i].position["real1D"][j] = np.clip(population[i].position["real1D"][j],
                                                                  self._min_range_of_real_variables,
                                                                  self._max_range_of_real_variables)

                    population[i + 1].position["real1D"][j] = np.clip(population[i + 1].position["real1D"][j],
                                                                      self._min_range_of_real_variables,
                                                                      self._max_range_of_real_variables)

            return population

        for type_of_variable in self._type_number_of_variables.keys():

            if type_of_variable == "binary1D":

                population_crossover = apply_crossover_binary1D(population_crossover)

            if type_of_variable == "real1D":

                population_crossover = apply_crossover_real1D(population_crossover)

        return population_crossover

    def _apply_mutation(self):

        population_mutation = [self._Individual() for _ in range(self._number_of_mutation)]

        def apply_mutation_binary1D(population):

            for i in range(len(population)):

                population[i].position["binary1D"] = {}

                if self._parents_selection_method == "roulette_wheel_selection":

                    parent_index = self._roulette_wheel_selection(self._population_main_probs)

                else:

                    parent_index = self._tournament_selection()

                for j in range(len(self._type_number_of_variables["binary1D"])):

                    parent = self._population_main[parent_index].position["binary1D"][j]

                    number_of_mutations = int(np.ceil(self._mutation_rate *
                                                      self._type_number_of_variables["binary1D"][j]))

                    mutated_cells = np.random.choice(range(self._type_number_of_variables["binary1D"][j]),
                                                     number_of_mutations, replace=False)

                    population[i].position["binary1D"][j] = copy.deepcopy(parent)

                    population[i].position["binary1D"][j][:, mutated_cells] = \
                    1 - population[i].position["binary1D"][j][:, mutated_cells]


            return population

        def apply_mutation_real1D(population):

            for i in range(len(population)):

                population[i].position["real1D"] = {}

                if self._parents_selection_method == "roulette_wheel_selection":

                    parent_index = self._roulette_wheel_selection(self._population_main_probs)

                else:

                    parent_index = self._tournament_selection()

                for j in range(len(self._type_number_of_variables["real1D"])):

                    parent = self._population_main[parent_index].position["real1D"][j]

                    number_of_mutation = int(np.ceil(self._mutation_rate * self._type_number_of_variables["real1D"][j]))

                    mutated_cells = [int(i) for i in
                                     np.random.choice(range(self._type_number_of_variables["real1D"][j]),
                                                      number_of_mutation, replace=False)]

                    population[i].position["real1D"][j] = copy.deepcopy(parent)

                    population[i].position["real1D"][j][0, mutated_cells] += \
                    self._gamma_for_mutation * (self._max_range_of_real_variables - self._min_range_of_real_variables) \
                    * np.random.randn(number_of_mutation)


                    population[i].position["real1D"][j] = np.clip(population[i].position["real1D"][j],
                                                                  self._min_range_of_real_variables,
                                                                  self._max_range_of_real_variables)
            return population

        for type_of_variable in self._type_number_of_variables.keys():

            if type_of_variable == "binary1D":

                population_mutation = apply_mutation_binary1D(population_mutation)

            if type_of_variable == "real1D":

                population_mutation = apply_mutation_real1D(population_mutation)

        return population_mutation

    def _merge(self):

        pop_main = copy.deepcopy(self._population_main)
        pop_crossover = copy.deepcopy(self._population_crossover)
        pop_mutation = copy.deepcopy(self._population_mutation)

        pop_new = []
        pop_new.extend(pop_main)
        pop_new.extend(pop_crossover)
        pop_new.extend(pop_mutation)

        return pop_new

    def _truncate(self):

        return self._population_main[:self._number_of_population]

    def run(self):

        tic = time.time()

        assert callable(self._cost_function), "cost function should be callable."

        assert self._min_range_of_real_variables < self._max_range_of_real_variables, "min range of real variables " \
                                                                                      "should be less than max " \
                                                                                      "range of real variables."
        assert self._min_range_of_integer_variables < self._max_range_of_integer_variables, \
        "min range of integer variables should be less than max range of integer variables."
        assert 0 <= self._crossover_percentage <= 1, "crossover percentage should be between 0 and 1."
        assert 0 <= self._mutation_percentage <= 1, "mutation percentage should be between 0 and 1."
        assert 0 <= self._mutation_rate <= 1, "mutation rate should be between 0 and 1."
        assert 0 <= self._gamma_for_crossover <= 1, "gamma for crossover should be between 0 and 1."
        assert 0 <= self._gamma_for_mutation <= 1, "gamma for mutation should be between 0 and 1."
        assert self._parents_selection_method in ["roulette_wheel_selection", "tournament_selection"], \
        "parent selection method can be either roulette wheel or tournament selection."
        assert self._selection_pressure >= 0, "selection pressure can not be negative."
        assert self._tournament_size >= 2, "tournament size should be greater or equal to 2."
        assert self._crossover_method in ["single_point", "double_point", "uniform"], \
        "crossover methods can be \"single point\", \"double point\" or \"uniform\". "
        assert self._yaxis in ["linear", "log"], "y axis for plotting can be \"linear\" or \"log\"."


        self._population_main = self._initialize_population()

        self._population_main = self._evaluate_population(self._population_main)

        self._population_main = self._sort(self._population_main)

        self._worst_cost = self._population_main[-1].cost

        for iter_main in range(self._max_iteration):

            self._population_main_probs = self._population_probabilities_calculations()

            self._population_crossover = self._apply_crossover()

            self._population_crossover = self._evaluate_population(self._population_crossover)

            self._population_mutation = self._apply_mutation()

            self._population_mutation = self._evaluate_population(self._population_mutation)

            self._population_main = self._merge()

            self._population_main = self._sort(self._population_main)

            self._population_main = self._truncate()

            self._best_cost.append(self._population_main[0].cost)

        toc = time.time()


        if self._plot_cost_function:

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))
            if self._yaxis == "linear":
                plt.plot(range(self._max_iteration), self._best_cost)
            if self._yaxis == "log":
                plt.semilogy(range(self._max_iteration), self._best_cost)
            plt.xlabel("Number of Iteration")
            plt.ylabel("Best Cost")
            plt.title("Objective Function Value Per Iteration Using Genetic Algorithm", fontweight="bold")
            plt.savefig("./figures/cost_function_ga.png")

        return self._population_main[0], toc - tic
