import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os


class PSO:

    class _Particle:

        def __init__(self):

            self.position = {}
            self.position_best = {}
            self.velocity = {}
            self.solution_parsed = {}
            self.cost = None
            self.cost_best = 1e20

    class _ParticleBest:

        def __init__(self):

            self.position = {}
            self.cost = 1e20
            self.solution_parsed = {}

    def __init__(self,
                 cost_function,
                 type_number_of_variables,
                 min_range_of_variables,
                 max_range_of_variables,
                 max_iteration=100,
                 number_of_particles=20,
                 inertia_rate=1,
                 inertia_damping_rate=1,
                 personal_learning_rate=2,
                 global_learning_rate=2,
                 tempo_limit=False,
                 plot_cost_function=False,
                 y_axis="linear",
                 mutation_for_real=False,
                 gamma_for_real_mutation=0.1,
                 mutation_for_permutation=False):

        self._cost_function = cost_function
        self._type_number_of_variables = type_number_of_variables
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._max_iteration = max_iteration
        self._number_of_particles = number_of_particles
        self._inertia_rate = inertia_rate
        self._inertia_damping_rate = inertia_damping_rate
        self._learning_rate_personal = personal_learning_rate
        self._learning_rate_global = global_learning_rate
        self._tempo_limit = tempo_limit
        self._tempo_max = 0.1 * (self._max_range_of_variables - self._min_range_of_variables)
        self._tempo_min = -self._tempo_max
        self._plot_cost_function = plot_cost_function
        self._y_axis = y_axis
        self._mutation_for_real = mutation_for_real
        self._gamma_for_real_mutation = gamma_for_real_mutation
        self._mutation_for_permutation = mutation_for_permutation
        self._particles = None
        self._particle_best = self._ParticleBest()
        self._best_cost = []
        self._mutated_particles = [self._Particle() for _ in range(self._number_of_particles)]
        self._mutated_particle_best = self._ParticleBest()

    def _initialize_particles(self):

        particles = [self._Particle() for _ in range(self._number_of_particles)]

        def initialize_particles_real1D(particle):

            for i in range(len(particle)):

                particle[i].position["real1D"] = {}
                particle[i].velocity["real1D"] = {}

                for j in range(len(self._type_number_of_variables["real1D"])):

                    particle[i].position["real1D"][j] = \
                    np.random.uniform(self._min_range_of_variables,
                                      self._max_range_of_variables, (1, self._type_number_of_variables["real1D"][j]))

                    particle[i].velocity["real1D"][j] = np.zeros_like(particle[i].position["real1D"][j])

            return particle

        def initialize_particles_real2D(particle):

            for i in range(len(particle)):

                particle[i].position["real2D"] = {}
                particle[i].velocity["real2D"] = {}

                for j in range(len(self._type_number_of_variables["real2D"])):

                    particle[i].position["real2D"][j] = \
                    np.random.uniform(self._min_range_of_variables,
                                      self._max_range_of_variables,
                                      self._type_number_of_variables["real2D"][j])

                    particle[i].velocity["real2D"][j] = np.zeros_like(particle[i].position["real2D"][j])

            return particle

        for type_of_variables in self._type_number_of_variables.keys():

            if type_of_variables == "real1D":

                particles = initialize_particles_real1D(particles)

            if type_of_variables == "real2D":

                particles = initialize_particles_real2D(particles)

        return particles

    def _evaluate_particles(self, particles):

        for i in range(len(particles)):

            particles[i].solution_parsed, \
            particles[i].cost = self._cost_function(particles[i].position)

            if particles[i].cost <= particles[i].cost_best:

                particles[i].position_best = copy.deepcopy(particles[i].position)
                particles[i].cost_best = copy.deepcopy(particles[i].cost)

                if particles[i].cost <= self._particle_best.cost:

                    self._particle_best.position = copy.deepcopy(particles[i].position)
                    self._particle_best.solution_parsed = copy.deepcopy(particles[i].solution_parsed)
                    self._particle_best.cost = copy.deepcopy(particles[i].cost)


        return particles

    def _update_velocity_and_positions(self):

        particles = copy.deepcopy(self._particles)

        def update_velocity_and_position_real1D_2D(particle):

            for i in range(len(particle)):

                for type_of_variable in self._type_number_of_variables.keys():

                    for j in range(len(self._type_number_of_variables[type_of_variable])):

                        r1 = np.random.uniform(0, 1, particle[i].position[type_of_variable][j].shape)
                        r2 = np.random.uniform(0, 1, particle[i].position[type_of_variable][j].shape)

                        particle[i].velocity[type_of_variable][j] = \
                        self._inertia_rate * particle[i].velocity[type_of_variable][j] + \
                        self._learning_rate_personal * \
                        np.multiply(r1, particle[i].position_best[type_of_variable][j] -
                                    particle[i].position[type_of_variable][j]) + \
                        self._learning_rate_global * \
                        np.multiply(r2, self._particle_best.position[type_of_variable][j] -
                                    particle[i].position[type_of_variable][j])

                        if self._tempo_limit:

                            particle[i].velocity[type_of_variable][j] = \
                            np.clip(particle[i].velocity[type_of_variable][j], self._tempo_min, self._tempo_max)

                        particle[i].position[type_of_variable][j] += particle[i].velocity[type_of_variable][j]

                        particle[i].position[type_of_variable][j] = np.clip(particle[i].position[type_of_variable][j],
                                                                             self._min_range_of_variables,
                                                                             self._max_range_of_variables)

                        if self._mutation_for_real:

                            self._mutated_particles[i].position[type_of_variable] = {}

                            if type_of_variable == "real1D":

                                alpha = np.random.uniform(-self._gamma_for_real_mutation,
                                                          1 + self._gamma_for_real_mutation,
                                                          (1, self._type_number_of_variables["real1D"][j]))

                            elif type_of_variable == "real2D":

                                alpha = np.random.uniform(-self._gamma_for_real_mutation,
                                                          1 + self._gamma_for_real_mutation,
                                                          self._type_number_of_variables["real2D"][j])

                            self._mutated_particles[i].position[type_of_variable][j] = \
                            np.multiply(alpha, particles[i].position[type_of_variable][j])

                            self._mutated_particles[i].position[type_of_variable][j] = \
                            np.clip(self._mutated_particles[i].position[type_of_variable][j],
                                    self._min_range_of_variables, self._max_range_of_variables)

                            self._mutated_particles[i].solution_parsed, \
                            self._mutated_particles[i].cost = self._cost_function(self._mutated_particles[i].position)

                            if self._mutated_particles[i].cost < particles[i].cost:

                                particles[i].position = copy.deepcopy(self._mutated_particles[i].position)
                                particles[i].cost = copy.deepcopy(self._mutated_particles[i].cost)
                                particles[i].solution_parsed = copy.deepcopy(self._mutated_particles[i].solution_parsed)

                                if particles[i].cost < particles[i].cost_best:

                                    particles[i].position_best = copy.deepcopy(particles[i].position)
                                    particles[i].cost_best = copy.deepcopy(particles[i].cost)

                            self._mutated_particle_best.position[type_of_variable] = {}

                            if type_of_variable == "real1D":

                                alpha = np.random.uniform(-self._gamma_for_real_mutation,
                                                          1 + self._gamma_for_real_mutation,
                                                          (1, self._type_number_of_variables["real1D"][j]))

                            elif type_of_variable == "real2D":

                                alpha = np.random.uniform(-self._gamma_for_real_mutation,
                                                          1 + self._gamma_for_real_mutation,
                                                          self._type_number_of_variables["real2D"][j])

                            self._mutated_particle_best.position[type_of_variable][j] = \
                            np.multiply(alpha, self._particle_best.position[type_of_variable][j])

                            self._mutated_particle_best.position[type_of_variable][j] = \
                            np.clip(self._mutated_particle_best.position[type_of_variable][j],
                                    self._min_range_of_variables,
                                    self._max_range_of_variables)

                            self._mutated_particle_best.solution_parsed, \
                            self._mutated_particle_best.cost = self._cost_function(self._mutated_particle_best.position)

                            if self._mutated_particle_best.cost < self._particle_best.cost:

                                self._particle_best = copy.deepcopy(self._mutated_particle_best)


                        if self._mutation_for_permutation:

                            def roulette_wheel_selection(probs):

                                random_number = np.random.rand()

                                probs_cum_sum = np.cumsum(probs)

                                return int(np.argwhere(random_number <= probs_cum_sum)[0][0])

                            def apply_mutation(p):

                                def swap(individual):

                                    indices = [int(_) for _ in
                                               np.random.choice(range(int(individual.shape[1])), 2, replace=False)]

                                    min_index, max_index = min(indices), max(indices)

                                    x = copy.deepcopy(individual)

                                    x[:, [min_index, max_index]] = individual[:, [max_index, min_index]]

                                    return x

                                def insertion(individual):

                                    indices = [int(_) for _ in
                                               np.random.choice(range(int(individual.shape[1])), 2, replace=False)]

                                    min_index, max_index = min(indices), max(indices)

                                    method = roulette_wheel_selection(np.random.dirichlet([0.5, 0.5]))

                                    if method == 0:

                                        x = np.concatenate((individual[:, :min_index + 1],
                                                            individual[:, max_index: max_index + 1],
                                                            individual[:, min_index + 1: max_index],
                                                            individual[:, max_index + 1:]), axis=1)

                                    else:

                                        x = np.concatenate((individual[:, :min_index],
                                                            individual[:, min_index + 1:max_index + 1],
                                                            individual[:, min_index: min_index + 1],
                                                            individual[:, max_index + 1:]), axis=1)

                                    return x

                                def reversion(individual):

                                    indices = [int(_) for _ in
                                               np.random.choice(range(int(individual.shape[1]) - 1), 2, replace=False)]

                                    min_index, max_index = min(indices), max(indices)

                                    x = np.concatenate((individual[:, :min_index],
                                                        np.flip(individual[:, min_index: max_index + 1]),
                                                        individual[:, max_index + 1:]), axis=1)

                                    return x

                                method_index = roulette_wheel_selection(np.random.dirichlet([0.3, 0.3, 0.4]))

                                if method_index == 0:

                                    return swap(p)

                                elif method_index == 1:

                                    return insertion(p)

                                else:

                                    return reversion(p)

                            self._mutated_particles[i].position[type_of_variable] = {}

                            a = np.argsort(particles[i].position[type_of_variable][j], axis=1)

                            mutated_a = apply_mutation(a)

                            self._mutated_particles[i].position[type_of_variable][j] = \
                            np.zeros_like(mutated_a, dtype=np.float64)

                            self._mutated_particles[i].position[type_of_variable][j][:, mutated_a[0, :]] = \
                            copy.deepcopy(particles[i].position[type_of_variable][j][:, a[0, :]])

                            self._mutated_particles[i].solution_parsed, \
                            self._mutated_particles[i].cost = self._cost_function(self._mutated_particles[i].position)

                            if self._mutated_particles[i].cost < particles[i].cost:

                                particles[i].position = copy.deepcopy(self._mutated_particles[i].position)
                                particles[i].cost = copy.deepcopy(self._mutated_particles[i].cost)
                                particles[i].solution_parsed = copy.deepcopy(self._mutated_particles[i].solution_parsed)

                                if particles[i].cost < particles[i].cost_best:
                                    particles[i].position_best = copy.deepcopy(particles[i].position)
                                    particles[i].cost_best = copy.deepcopy(particles[i].cost)

                            self._mutated_particle_best.position[type_of_variable] = {}

                            a = np.argsort(self._particle_best.position[type_of_variable][j], axis=1)

                            mutated_a = apply_mutation(a)

                            self._mutated_particle_best.position[type_of_variable][j] = \
                            np.zeros_like(mutated_a, dtype=np.float64)

                            self._mutated_particle_best.position[type_of_variable][j][:, mutated_a[0, :]] = \
                            copy.deepcopy(self._particle_best.position[type_of_variable][j][:, a[0, :]])

                            self._mutated_particle_best.solution_parsed, \
                            self._mutated_particle_best.cost = self._cost_function(self._mutated_particle_best.position)

                            if self._mutated_particle_best.cost < self._particle_best.cost:

                                self._particle_best = copy.deepcopy(self._mutated_particle_best)

            return particle

        if "real1D" in self._type_number_of_variables.keys() or "real2D" in self._type_number_of_variables.keys():

            particles = update_velocity_and_position_real1D_2D(particles)

        return particles

    def run(self):

        tic = time.time()

        assert callable(self._cost_function), "cost function should be callable."
        assert self._min_range_of_variables < self._max_range_of_variables, \
        "min range of variables should be less than max range of variables."

        self._particles = self._initialize_particles()

        self._particles = self._evaluate_particles(self._particles)

        for iter_main in range(self._max_iteration):

            self._particles = self._update_velocity_and_positions()

            self._particles = self._evaluate_particles(self._particles)

            self._best_cost.append(self._particle_best.cost)

            self._inertia_rate *= self._inertia_damping_rate

        toc = time.time()

        if self._plot_cost_function:

            os.makedirs("./figures", exist_ok=True)

            plt.figure(dpi=300, figsize=(10, 6))
            if self._y_axis == "linear":
                plt.plot(range(self._max_iteration), self._best_cost)
            if self._y_axis == "log":
                plt.semilogy(range(self._max_iteration), self._best_cost)
            plt.xlabel("Number of Iteration")
            plt.ylabel("Best Cost")
            plt.title("Objective Function Value Per Iteration Using Particle Swarm Optimization", fontweight="bold")
            plt.savefig("./figures/cost_function_pso.png")

        return self._particle_best, toc - tic
