#evolution process
"""
Use an evolutionary strategy to find a solutions that can derive the reference (training data) from the inputs. The
first generation is randomly generated. The next generations are based on the previous generations. Due to the
evolutionary process future generations should perform better.

Chromosome/individual refers to a list of integers the represents a possible solution.
Population a list of individuals.
Generation a population at a certain moment. Each generation produces the next generation.
"""
import cgp
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import matplotlib.animation as animation
from matplotlib import style
import time


class Graph:
    """
    Class that handles the drawing of the graph
    """

    def __init__(self, interval):
        """
        Create graph class
        :param interval: every interval generations will be plotted
        """
        self.interval = interval
        self.xs = []
        self.ys = []
        for i in range(10):
            self.ys.append([])
        self.ax = []
        self.fig = plt.figure()
        plt.interactive(False)
        for i in range(len(self.ys)):
            self.ax.append(self.fig.add_subplot(111))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def tick(self):
        #  for i in range(len(ys)):
        #      ax[i].clear()
        for i in range(len(self.ys)):
            self.ax[i].plot(self.xs, self.ys[i])
        self.fig.canvas.draw()
        plt.pause(0.001)


class GCPEvolver:
    """
    Class that handles the evolution process
    """
    def __init__(self, half_population=250, nr_nodes=10, mutation_chance=0.04, max_score=10000, nr_features=9):
        """
        Create a class that can be used to evolve a solution.
        :param half_population: size of half the population
        :param nr_nodes: number of nodes that should be used in each solution
        :param mutation_chance: Chance that an element mutates
        :param max_score: Stops when this score is reached
        :param nr_features: Number of features that will be used
        """
        self.half_population = half_population  # 0.5 * the number of chromosomes in a single generation
        self.nr_nodes = nr_nodes  # number of nodes in the chromosomes
        self.mutation_chance = mutation_chance  # (0-1) chance that an element gets a random value.
        self.target_fitness_score = max_score  # stop when error is smaller than this value.
        self.node_size = 3  # size of a node in the chromosome

        self.population_size = 2 * self.half_population  # + 1 # total population size, +1 for the best previous solution
        self.nr_features = nr_features
        self.graph = None #Graph(10)

        self.openCLExecutor = None

    def calculate_error(self, calculated, reference):  # NOT USED FOR GPU
        """
        Calculate the fitness score. It compare the calculated to the desired value (reference).
        :param calculated: The calculated value.
        :param reference: The reference/desired value.
        :return: The fitness score for thes calculated and reference values.
        """
        a = calculated[0] - reference[0]
        b = calculated[1] - reference[1]
        return math.sqrt(a * a + b * b)

    def diversity(self, population):
        """
        Give the population a diversity score. Thus how diverse the chromosomes are. Is used for debugging.
        :param population: A list of chromosomes
        :return: A diversity score for the given population.
        """
        dim = np.shape(population)
        k = 0
        for j in range(dim[0]):
            if (population[j] is None):
                print(j + " " + population[j])
            zerosoneslist=cgp.createListnodes(population[j], self.nr_features)
            for k in range(dim[1]):
                if type(population[j][k]) is int:
                    #print(k)
                    population[j][k]*=cgp.zerosoneslist[k]
            symbols = 100 * [0]
            # For each position count the number of different symbols.
            for i in range(dim[0]):
                if type(population[i][j]) is int and symbols[population[i][j]] == 0:
                    k += 1
                    symbols[population[i][j]] = 1
        return k / dim[1] / dim[0]

    def average(self, fitness_list):
        """
        Calculates the average fitness score for the fitness_list.
        :param fitness_list: A list of tuples where the 2 tuple element contains the fitness score.
        :return: The average fitness score.
        """
        total = 0
        for i in range(len(fitness_list)):
            total += fitness_list[i][1]
        return total / len(fitness_list)


    def create_base_population(self, size):
        """
        Create a base population of size number of chromosomes.
        :param size: Number of chromosomes that should be created.
        :param random_range: the range that should be used for the random number
        :return: A randomly generated population.
        """
        population = size * [None]
        for i in range(size):
            population[i] = cgp.translate(self.nr_features,
                                          np.random.randint(0,
                                                            self.nr_nodes + self.nr_features,
                                                            self.nr_nodes * self.node_size))
        return population

    def test_population(self, population, features, reference):  # NOT USED FOR GPU
        """
        Calculates for each individual in the population the fitness score using the given features.
        :param population: A list if chromosomes/individuals
        :param features: The features that should be used to calculate the fitness score
        :param reference: The references values to which the outputs must be compared to.
        :return: A list of tuples. In the tuples the first item is the chromosome id in the population and the second
        element is the fitness score.
        """
        fitness_list = self.population_size * [None]

        for i in range(self.population_size):
            fitness = 0.0
            for j in range(len(features)):
                output = cgp(features[j], population[i])
                fitness += self.calculate_fitness(output, reference[j])
            fitness_list[i] = (i, fitness / len(features))
        return fitness_list

    def get_index(self):
        index = self.population_size
        while index > self.population_size - 1:
            index = math.floor(abs(random.normalvariate(0, 0.05 * self.population_size)))
        return index

    def create_next_generation(self, fitness_list, population, best_solution):
        """
        Create the next generation based on the current population and their fitness score. The next generation is
        created using crossover and mutation.
        :param fitness_list: A sorted list of tuples. A first tuple element refers to the id in the population and the
        second element is the fitness score. The list is sorted on the fitness score.
        :param population: A list of chromosomes.
        :param best_solution: The best solution. This solution is always passed to the next generation.
        :return: A new population.
        """
        crossover_selection_list = []  # create a list that can be used to determine select parents for the next generation.
        for i in range(self.population_size):
            # add the chromosome index (population_size - i) times to the list.
            crossover_selection_list += (self.population_size - i) * [fitness_list[i][0]]

        next_generation = self.population_size * [None]
        for i in range(self.half_population):
            # select parents
            parent_id_1 = self.get_index()
            parent_id_2 = self.get_index()

            parent_1 = population[fitness_list[parent_id_1][0]]
            parent_2 = population[fitness_list[parent_id_2][0]]

            # get crossover location
            cross_location = random.randint(1, self.nr_nodes * self.node_size)

            child_1 = self.nr_nodes * self.node_size * [None]
            child_2 = self.nr_nodes * self.node_size * [None]
            for j in range(self.nr_nodes * self.node_size):
                # apply crossover
                if j < cross_location:
                    child_1[j] = parent_1[j]
                    child_2[j] = parent_2[j]
                else:
                    child_1[j] = parent_2[j]
                    child_2[j] = parent_1[j]

                # apply mutation
                if random.uniform(0, 1) < self.mutation_chance:
                    child_1[j] = cgp.translate_item(self.nr_features, j, random.randint(0, 99))

                if random.uniform(0, 1) < self.mutation_chance:
                    child_2[j] = cgp.translate_item(self.nr_features, j, random.randint(0, 99))

            # add children to the next generation.
            next_generation[2 * i] = child_1
            next_generation[2 * i + 1] = child_2

        # add best solution to the next generation
        #next_generation[population_size - 1] = best_solution
        return next_generation

    def predict(self, solution, features):
        """
        Use solution to create the result for the input features
        :param solution: solution that should be used to calculate the result
        :param features: input features
        :return: the values predicted by solution
        """
        return self.openCLExecutor.predict(solution, features)

    def evolve(self, features, reference):
        """
        Use an evolutionary strategy to find a solutions that can derive the reference (training data) from features.
        :param features: The input data for the model.
        :param reference: The reference data.
        :return: The best found chromosome.
        """
        dim_features= np.shape(features)
        population = self.create_base_population(self.population_size)
        best_fitness_score = -1
        best_solution = population[0]  # best seen solution

        # create object that can exectute the code on the gpu
        self.openCLExecutor = OpenCLExecutor(features, reference, self.nr_features, self.nr_nodes)

        # Simulate the generations in the evolution process.
        try:
            print("Start evolution, press ctrl-c to stop")
            for g in range(30000):
                # check if done
                if best_fitness_score != -1 and best_fitness_score > self.target_fitness_score:
                    break

                fitness_list = self.openCLExecutor.execute(population)
             #   fitness_list = test_population(population, features, reference) #  None gpu implementation (probably borken)
                fitness_list.sort(key=lambda i: i[1], reverse=True)  # sort the tuples on the second element, thus the fitness score
                # Check if a new best solution has been found.
                if best_fitness_score == -1 or fitness_list[0][1] > best_fitness_score:
                        best_solution = population[fitness_list[0][0]]  # store the best solution
                        best_fitness_score = fitness_list[0][1]  # store the error

                        # print for debugging
                        print( population[fitness_list[0][0]])
                        print(best_fitness_score)

                        # check if it can stop
                        if best_fitness_score > self.target_fitness_score:
                            print("Done")
                            break

                #  Show graph, makes it slower
                if self.graph is not None and g % self.graph.interval == 0:
                    self.graph.xs.append(g)
                    interval = math.floor(self.population_size / len(self.graph.ys))
                    for i in range(len(self.graph.ys)):
                        self.graph.ys[i].append(fitness_list[interval * i][1])
                    self.graph.tick()

                print("Generation ", g, ", smallest error: ", fitness_list[0][1], ", error median: ",
                      fitness_list[self.half_population][1], ", diversity: ", "") #diversity(population))

                population = self.create_next_generation(fitness_list, population, best_solution)
        # Stop when ctrl-c is pressed
        except KeyboardInterrupt:
            print("Evolution stopped, processing results...")
            pass

        return best_solution


class OpenCLExecutor:
    def __init__(self, features, reference, nr_features, nr_nodes):
        self.nr_points = len(features)
        self.features_np = np.asarray(features, dtype=np.float32)
        self.reference_np = np.asarray(reference, dtype=np.float32)
        self.result_np = np.empty(self.nr_points, dtype=np.float32)

        self.ctx = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

        self.features_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.features_np)
        self.reference_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.reference_np)
        self.result_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.result_np.nbytes)

        self.program_fitness = cl.Program(self.ctx, """
        __kernel void calculate_fitness(
            __global const float *r_g, __global const float *f_g, __global float *res_g, __global int *program)
        {
          int nr_features = """ + str(nr_features) + """;
          int nr_nodes = """ + str(nr_nodes) + """;
          int gid = get_global_id(0);
          int offset = gid * """ + str(nr_features) + """;
          float inputs[""" + str(nr_features + nr_nodes) + """] ;
          for (int i = 0; i < nr_features; i++) {
              inputs[i] = f_g[offset + i];
          }
          for (int i = 0; i < nr_nodes; i++) {
              int id1 = program[i * 3];
              int id2 = program[i * 3 + 1];
              int op  = program[i * 3 + 2];
              float i1 = inputs[id1];
              float i2 = inputs[id2];
              if(op == 0)
                  inputs[i + nr_features] = i1 + i2;
              if(op == 1)
                  inputs[i + nr_features] = i1 - i2;
              if(op == 2)
                  inputs[i + nr_features] = i1 * i2;
              if(op == 3) {
                float safe_offset = (i2 > 0) ? FLT_EPSILON : -FLT_EPSILON;
                inputs[i + nr_features] = i1 / (i2 + safe_offset);    
              }                  
          }
          float result1= inputs[nr_features + nr_nodes -1] - r_g[2 * gid];
          float result2= inputs[nr_features + nr_nodes -2] - r_g[2 * gid + 1];
          float result = sqrt(result1 * result1 + result2 * result2);
          //res_g[gid] = 1 / (result + 0.02);
          res_g[gid] = -result;
        }
        """).build()

        self.program_predict = cl.Program(self.ctx, """
        __kernel void predict(
           __global const float *f_g, __global float *res_g, __global int *program)
        {
          int nr_features = """ + str(nr_features) + """;
          int nr_nodes = """ + str(nr_nodes) + """;
          int gid = get_global_id(0);
          int offset = gid * """ + str(nr_features) + """;
          float inputs[""" + str(nr_features + nr_nodes) + """] ;
          for (int i = 0; i < nr_features; i++) {
              inputs[i] = f_g[offset + i];
          }
          for (int i = 0; i < nr_nodes; i++) {
              int id1 = program[i * 3];
              int id2 = program[i * 3 + 1];
              int op  = program[i * 3 + 2];
              float i1 = inputs[id1];
              float i2 = inputs[id2];
              if(op == 0)
                  inputs[i + nr_features] = i1 + i2;
              if(op == 1)
                  inputs[i + nr_features] = i1 - i2;
              if(op == 2)
                  inputs[i + nr_features] = i1 * i2;
              if(op == 3) {
                float safe_offset = (i2 > 0) ? FLT_EPSILON : -FLT_EPSILON;
                inputs[i + nr_features] = i1 / (i2 + safe_offset);    
              }                  
          }
          res_g[gid * 2] =  inputs[nr_features + nr_nodes -1];
          res_g[gid * 2 + 2] = inputs[nr_features + nr_nodes -2];
        }
        """).build()

    def execute(self, population):
        population_size = len(population)
        fitness_list = population_size * [None]
        for i in range(population_size):
            translated_np = np.asarray(cgp.translate_operation_to_ints(population[i]), dtype=np.int32)
            translated_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=translated_np)
            self.program_fitness.calculate_fitness(self.queue,
                                                   self.result_np.shape,
                                                   None, self.reference_g,
                                                   self.features_g,
                                                   self.result_g, translated_g)
            cl.enqueue_copy(self.queue, self.result_np, self.result_g)
            fitness_list[i] = (i, np.sum(self.result_np) / self.nr_points)
        return fitness_list

    def predict(self, solution, features):
        result_predict_np = np.empty((len(features), 2))
        result_predict_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, result_predict_np.nbytes)

        translated_np = np.asarray(cgp.translate_operation_to_ints(solution), dtype=np.int32)
        translated_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=translated_np)
        f2_np = np.asarray(features, dtype=np.float32)
        f2_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=f2_np)

        self.program_predict.predict(self.queue, result_predict_np.shape, None, f2_g, result_predict_g, translated_g)
        cl.enqueue_copy(self.queue, result_predict_np, result_predict_g)
        return result_predict_np


"""
   Generate test data
   """
ref = []
f = []
x = np.linspace(-1, 1, 1000)
for i in range(len(x)):
   
    f += [[x[i]]]
    #res = 2 * x - 3 * y + 4 * z - u * x
    res = math.tanh(x[i])
    ref += [[res, res]]


evolver = GCPEvolver(half_population=250, nr_nodes=50, mutation_chance=0.01, max_score=10000, nr_features=1)

result = evolver.evolve(f, ref)

# Plots
tra = cgp.complete_translate(result, evolver.nr_features, evolver.nr_nodes)
print (tra[evolver.nr_nodes+evolver.nr_features -1])
print (tra[evolver.nr_nodes+evolver.nr_features -2])