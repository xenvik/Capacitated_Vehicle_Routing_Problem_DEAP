import os
import io
import pandas as pd
import numpy as np
import random
import glob
import csv
import matplotlib.pyplot as plt
from csv import DictWriter
from json import load, dump
from deap import base, creator, tools, algorithms, benchmarks

#####################################################################################
## DEAP CAPACITATED VEHICLE ROUTING PROBLEM MODEL FOR IMPLEMENTATION OF NSGA-2 ALGORITHM

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname('Input_Data.txt')))


## Initialising the problem, giving input file in json format
"""
Parameters: json file path
Returns: file object or else NoneType
"""
def input_instance(json_file):
    if os.path.exists(path=json_file):
        with io.open(json_file, 'rt', newline='') as file_object:
            return load(file_object)
    return None


## Individual route sub-divided into subroute and assigned to vehicle
"""
Parameters: Order of clients as an individual route, file object
Returns: Subroutes assigned to each vechicle appended into a route
"""
def route_subroute(individual, instance):
    route = []
    sub_route = []
    vehicle_load = 0
    previous_client_num = 0
    vehicle_capacity = instance['vehicle_capacity']

    for client_num in individual:
        demand = instance[f"client_{client_num}"]["demand"]
        vehicle_load_updated = vehicle_load + demand

        if (vehicle_load_updated <= vehicle_capacity):
            sub_route.append(client_num)
            vehicle_load = vehicle_load_updated
        else:
            route.append(sub_route)
            sub_route = [client_num]
            vehicle_load = demand

        previous_client_num = client_num

    if sub_route != []:
        route.append(sub_route)

    return route

## Getting the individual route and subroute printed
"""
Parameters: Individual route
Returns: Prints route and subroute
"""
def show_route(route, merge=False):
    route_str = '0'
    sub_route_count = 0
    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        for client_num in sub_route:
            sub_route_str = f'{sub_route_str} - {client_num}'
            route_str = f'{route_str} - {client_num}'
        sub_route_str = f'{sub_route_str} - 0'
        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
        route_str = f'{route_str} - 0'
    if merge:
        print(route_str)


## For an individual route, number of vehicles required
"""
Parameters: Individual route, file object
Returns: Number of vechiles required for the individual route
"""
def vehicles_required(individual, instance):

    # Updating route to subroutes as per demand
    route_updated = route_subroute(individual, instance)
    num_of_vehicles = len(route_updated)
    return num_of_vehicles


## Calculating cost of an individual route
"""
Parameters: Individual route, file object, cost per unit for the route (assumed=1)
Returns: Total cost for the route taken by all the vehicles
"""
def route_cost(individual, instance, unit_cost=1):

    total_cost = 0
    route_updated = route_subroute(individual, instance)

    for sub_route in route_updated:
        sub_route_distance = 0
        previous_client_num = 0

        for client_num in sub_route:
            # Distance from previous client to next in the subroute
            distance = instance["distance_matrix"][previous_client_num][client_num]
            sub_route_distance += distance
            previous_client_num = client_num

        # Adding to subroute cost from last client to depot
        sub_route_distance = sub_route_distance + instance["distance_matrix"][previous_client_num][0]

        # Subroute cost
        sub_route_cost = unit_cost * sub_route_distance

        # Total Cost
        total_cost = total_cost + sub_route_cost

    return total_cost


## Fitness of individual route
"""
Parameters: individual route, file object, per unit cost of route
Returns: Tuple of (Number of vechicles required, Total route cost
"""
def evaluate_fitness(individual, instance, unit_cost):

    vehicles = vehicles_required(individual, instance)
    individual_route_cost = route_cost(individual, instance, unit_cost)

    return (vehicles, individual_route_cost)


## Crossover method with ordering
"""
Parameters: individual routes to crossover
Returns: Tuple of (Number of vechicles required, Total route cost)
"""
def crossover_order(cross_ind1, cross_ind2):
    #  If sequence does not contain 0, this throws error
    #  Modify inputs and outputs
    ind1 = [x - 1 for x in cross_ind1]
    ind2 = [x - 1 for x in cross_ind2]
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    cuts1, cuts2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            cuts1[ind2[i]] = False
            cuts2[ind1[i]] = False

    # Keep original values before crossover ordering
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not cuts1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not cuts2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swapping all including a and b
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    # Reclaiming original input
    ind1 = [x + 1 for x in ind1]
    ind2 = [x + 1 for x in ind2]
    return ind1, ind2

## Mutation with shuffling
"""
Parameters: Individual route, probability of mutation
Returns: Mutated individual
"""
def ind_mutation(individual, indpb):
    size = len(individual)
    
    for i in range(size):
        if random.random() < indpb:
            shuffle_index = random.randint(0, size - 2)
            if shuffle_index >= i:
                shuffle_index += 1
            individual[i], individual[shuffle_index] = \
                individual[shuffle_index], individual[i]

    return individual,


## Making a logbook to log each generation of individual routes data
"""
Parameters: None
Returns: Tuple of logbook and statistics objects
"""
def statistics_objects():
    # Create stats and logbook objects using DEAP

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Logging
    logbook = tools.Logbook()
    logbook.header = "Generation", "evals", "avg", "std", "min", "max", "best_individual", "fitness_best_individual"
    return logbook, stats

## Recording Stats for selected individuals
"""
Parameters: Individual for which fitness is calculated, logbook object, population, stats object
Returns: None, prints the logs using stream
"""
def record_stats(invalid_ind, logbook, pop, stats, gen):
    record = stats.compile(pop)
    best_individual = tools.selBest(pop, 1)[0]
    record["best_individual"] = best_individual
    record["fitness_best_individual"] = best_individual.fitness
    logbook.record(Generation=gen, evals=len(invalid_ind), **record)
    print(logbook.stream)


## Exporting CSV files
"""
Parameters: CSV file name, logbook
Returns: Exception with I/O error
"""
def export_csv(csv_file_name, logbook):
    csv_columns = logbook[0].keys()
    csv_path = os.path.join(BASE_DIR, "results", csv_file_name)
    try:
        with open(csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in logbook:
                writer.writerow(data)
    except IOError:
        print("I/O error")

## NSGA-2 Algorithm Implementation
"""
Parameters: CSV file name, logbook
Returns: Exception with I/O error
"""

class nsga2_algo(object):

    def __init__(self):
        self.json_instance = input_instance('./data/json/Input_Data.json')
        self.ind_size = self.json_instance['Number_of_clients']
        self.pop_size = 500
        self.num_gen = 200
        self.cross_prob = 0.80
        self.mut_prob = 0.02
        self.toolbox = base.Toolbox()
        self.creators()
        self.logbook, self.stats = statistics_objects()

    # Using DEAP model tools for Evaluation and Generating Population as per NSGA-2 algorithm
    def creators(self):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # Sampling
        self.toolbox.register('indexes', random.sample, range(1, self.ind_size + 1), self.ind_size)

        # Generating individual and population from individual route
        self.toolbox.register('individual', tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Evaluate function for fitness check
        self.toolbox.register('evaluate', evaluate_fitness, instance=self.json_instance, unit_cost=1)

        # Selection
        self.toolbox.register("select", tools.selNSGA2)

        # Crossover
        self.toolbox.register("mate", crossover_order)

        # Mutation
        self.toolbox.register("mutate", ind_mutation, indpb=self.mut_prob)

    # Population Fitness for Individuals
    def population_fitness(self):
        self.pop = self.toolbox.population(n=self.pop_size)
        self.invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self.fitnesses = list(map(self.toolbox.evaluate, self.invalid_ind))

        for ind, fit in zip(self.invalid_ind, self.fitnesses):
            ind.fitness.values = fit

        self.pop = self.toolbox.select(self.pop, len(self.pop))

        record_stats(self.invalid_ind, self.logbook, self.pop, self.stats, gen=0)

    # Generation population using genetic algorithm
    def generations(self):
        # Running Genetic algorithm
        for gen in range(self.num_gen):
            print(f"{20 * '#'} Currently Evaluating {gen} Generation {20 * '#'}")

            # Selecting individuals and offsprings from the population using TournamentDCD
            self.offspring = tools.selTournamentDCD(self.pop, len(self.pop))
            self.offspring = [self.toolbox.clone(ind) for ind in self.offspring]

            # Crossover and mutation
            for ind1, ind2 in zip(self.offspring[::2], self.offspring[1::2]):
                if random.random() <= self.cross_prob:
                    self.toolbox.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)

            # Fitness for all the invalid individuals in offspring
            self.invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            self.fitnesses = self.toolbox.map(self.toolbox.evaluate, self.invalid_ind)
            for ind, fit in zip(self.invalid_ind, self.fitnesses):
                ind.fitness.values = fit

            # Population added with offsprings
            self.pop = self.toolbox.select(self.pop + self.offspring, self.pop_size)

            # Record stats of this generation
            record_stats(self.invalid_ind, self.logbook, self.pop, self.stats, gen + 1)

        print(f"{20 * '#'} End of Generations {20 * '#'} ")

    # Selecting the best generations
    def select_best_ind(self):
        self.best_individual = tools.selBest(self.pop, 1)[0]

        print(f"Best individual is {self.best_individual}")
        print(f"Number of vechicles required are "
              f"{self.best_individual.fitness.values[0]}")
        print(f"Cost required for the transportation is "
              f"{self.best_individual.fitness.values[1]}")

        # Route of the best individual
        show_route(route_subroute(self.best_individual, self.json_instance))

    # Expost CSV name
    def export_csv_name(self):
        csv_file_name = f"{self.json_instance['instance_name']}_" \
                        f"Population{self.pop_size}_Crossover{self.cross_prob}" \
                        f"_Mutation{self.mut_prob}_Generation{self.num_gen}.csv"
        export_csv(csv_file_name, self.logbook)

    # Run All
    def run_all(self):
        self.population_fitness()
        self.generations()
        self.select_best_ind()
        self.export_csv_name()

##################################################################################################
## PLOTTING RESULTS

def results_path():
    allpaths = glob.glob("./results/*.csv")
    allpaths = [i.replace("\\","/") for i in allpaths]
    csv_files = [eachpath.split("/")[-1] for eachpath in allpaths]
    return allpaths, csv_files


def load_csv(csv_file_path):
    instance = pd.read_csv(csv_file_path)
    return instance


def required_results(csv_file_path):
    loaded_result = load_csv(csv_file_path)
    min_column = loaded_result['min']
    gen_column = loaded_result['Generation']

    def clean_row(inp):
        out = inp.replace("[","").replace("]","").strip().split(" ")
        return out

    min_dist = [float(clean_row(i)[-1]) for i in min_column]
    min_vehicles = [float(clean_row(i)[0]) for i in min_column]
    return min_dist, gen_column


def Fitness_plot(csv_file_path):
    distances, generations = required_results(csv_file_path)
    csv_title = csv_file_path.split("/")[-1][:-4]
    fig = plt.figure(figsize=(10, 10))
    plt.plot(generations, distances)
    plt.xlabel("Generations")
    plt.ylabel("Min distance")
    plt.title(csv_title)
    plt.savefig(f"./figures/Fitness_{csv_title}.png")
    plt.show()


def Fitness_all_plots():
    allpaths, csv_files = results_path()

    # Plotting all
    for eachpath in allpaths:
        Fitness_plot(eachpath)


## Loading locations and clients to dataframe
def coordinate_dataframe(json_instance):
    num_of_client = json_instance['Number_of_clients']
    # Client coordinates
    client_list = [i for i in range(1, num_of_client + 1)]
    x_coord_client = [json_instance[f'client_{i}']['coordinates']['x'] for i in client_list]
    y_coord_client = [json_instance[f'client_{i}']['coordinates']['y'] for i in client_list]
    # Depot coordinates
    depot_x = [json_instance['depart']['coordinates']['x']]
    depot_y = [json_instance['depart']['coordinates']['y']]
    # Depot details
    client_list = [0] + client_list
    x_coord_client = depot_x + x_coord_client
    y_coord_client = depot_y + y_coord_client
    df = pd.DataFrame({"X": x_coord_client,
                       "Y": y_coord_client,
                       "client_list": client_list
                       })
    return df


def plot_subroute(subroute, dfcoord,color):
    totalSubroute = [0]+subroute+[0]
    subroutelen = len(subroute)
    for i in range(subroutelen+1):
        firstcust = totalSubroute[0]
        secondcust = totalSubroute[1]
        plt.plot([dfcoord.X[firstcust], dfcoord.X[secondcust]],
                 [dfcoord.Y[firstcust], dfcoord.Y[secondcust]], c=color)
        totalSubroute.pop(0)


def plot_route(route, csv_title):
    # Loading the instance
    json_instance = input_instance('./data/json/Input_Data.json')

    subroutes = route_subroute(route, json_instance)
    colorslist = ["blue","green","red","cyan","magenta","yellow","black","gold"]
    colorindex = 0

    # getting df
    dfcoord = coordinate_dataframe(json_instance)

    # Plotting scatter
    plt.figure(figsize=(10, 10))

    for i in range(dfcoord.shape[0]):
        if i == 0:
            plt.scatter(dfcoord.X[i], dfcoord.Y[i], c='blue', s=200)
            plt.text(dfcoord.X[i], dfcoord.Y[i], "depot", fontsize=12)
        else:
            plt.scatter(dfcoord.X[i], dfcoord.Y[i], c='green', s=200)
            plt.text(dfcoord.X[i], dfcoord.Y[i], f'{i}', fontsize=12)

    # Plotting routes
    for route in subroutes:
        plot_subroute(route, dfcoord, color=colorslist[colorindex])
        colorindex += 1

    # Adding labels, Title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(csv_title)
    plt.show()
    plt.savefig(f"./figures/Route_{csv_title}.png")

'''
if __name__ == "__main__":
    sample_route = [1, 2, 4, 25, 24, 22, 23, 17, 13, 10, 15, 19, 18, 12, 14, 16, 11, 9, 6, 8, 7, 3, 5, 21, 20]
    plot_route(sample_route,"Sample_Title")
    plt.show()
'''

##############################################################################################
## UNIT TESTING FOR THE FUNCTIONS USED

def testcosts():
    # Sample instance
    test_instance = input_instance('./data/json/Input_Data.json')

    # Sample individual
    sample_individual = [19, 5, 24, 7, 16, 23, 22, 2, 12, 8, 20, 25, 21, 18, 11, 15, 1, 14, 17, 6, 4, 13, 10, 3, 9]

    # Sample individual 2
    sample_ind_2 = random.sample(sample_individual, len(sample_individual))
    print(f"Sample individual is {sample_individual}")
    print(f"Sample individual 2 is {sample_ind_2}")

    # Cost of Individual route
    print(f"Sample individual cost is {route_cost(sample_individual, test_instance, 1)}")
    print(f"Sample individual 2 cost is {route_cost(sample_ind_2, test_instance, 1)}")

    # Fitness of Individual route
    print(f"Sample individual fitness is {evaluate_fitness(sample_individual, test_instance, 1)}")
    print(f"Sample individual 2 fitness is {evaluate_fitness(sample_ind_2, test_instance, 1)}")


def testroutes():
    # Sample instance
    test_instance = input_instance('./data/json/Input_Data.json')

    # Sample individual
    sample_individual = [19, 5, 24, 7, 16, 23, 22, 2, 12, 8, 20, 25, 21, 18, 11, 15, 1, 14, 17, 6, 4, 13, 10, 3, 9]
    best_ind_300_gen = [16, 14, 12, 10, 15, 17, 21, 23, 11, 9, 8, 20, 18, 19, 13, 22, 25, 24, 5, 3, 4, 6, 7, 1, 2]

    # Sample individual 2
    sample_ind_2 = random.sample(sample_individual, len(sample_individual))
    print(f"Sample individual is {sample_individual}")
    print(f"Sample individual 2 is {sample_ind_2}")
    print(f"Best individual 300 generations is {best_ind_300_gen}")

    # Getting routes
    print(f"Subroutes for first sample individual is {route_subroute(sample_individual, test_instance)}")
    print(f"Subroutes for second sample indivudal is {route_subroute(sample_ind_2, test_instance)}")
    print(f"Subroutes for best sample indivudal is {route_subroute(best_ind_300_gen, test_instance)}")

    # Getting num of vehicles
    print(f"Vehicles for sample individual {vehicles_required(sample_individual, test_instance)}")
    print(f"Vehicles for second sample individual {vehicles_required(sample_ind_2, test_instance)}")
    print(f"Vehicles for best sample individual {vehicles_required(best_ind_300_gen, test_instance)}")


def testcrossover():
    ind1 = [3, 2, 5, 1, 6, 9, 8, 7, 4]
    ind2 = [7, 3, 6, 1, 9, 2, 4, 5, 8]
    anotherind1 = [16, 14, 12, 7, 4, 2, 1, 13, 15, 8, 9, 6, 3, 5, 17, 18, 19, 11, 10, 21, 22, 23, 25, 24, 20]
    anotherind2 = [21, 22, 23, 25, 16, 14, 12, 7, 4, 2, 1, 13, 15, 8, 9, 6, 3, 5, 17, 18, 19, 11, 10, 24, 20]

    newind7, newind8 = crossover_order(ind1, ind2)
    newind9, newind10 = crossover_order(anotherind1, anotherind2)

    print(f"InpInd1 is {ind1}")
    print(f"InpInd2 is {ind2}")
    print(f"newind7 is {newind7}")
    print(f"newind8 is {newind8}")
    print(f"newind9 is {newind9}")
    print(f"newind10 is {newind10}")


def testmutation():
    ind1 = [3, 2, 5, 1, 6, 9, 8, 7, 4]
    mut1 = ind_mutation(ind1)

    print(f"Given individual is {ind1}")
    print(f"Mutation from first method {mut1}")