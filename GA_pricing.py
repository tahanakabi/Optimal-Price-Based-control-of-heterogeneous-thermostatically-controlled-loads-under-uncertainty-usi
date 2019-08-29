import random
import time
import numpy as np
from lstm_test import predict
from keras.models import load_model
import pickle
import pandas as pd
from matplotlib import pyplot
from fuzzylogicTCL import fuzzy, tippings
from multiprocessing import Pool
#Global Variables
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

horizon = 6
users_num = 30

# horizons = dataframe.shape[0]//horizon
parameters = pd.read_csv('parameters.csv', header = None)






fixed_cost = 0
quadratic_price = .01
R_penalty = 1000.0

# load_max = np.ones(shape=horizon)*75.0
overload_penalty = 100

def lModel(i):
    model_name = 'lstm_model'+str(i)+'.h5'
    print(model_name+' loaded successfuly')
    return load_model(model_name)

def frange(start, stop=None, step=None):
    #Use float number in range() function
    # if stop and step argument is null set start=0.0 and step = 1.0
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step




def create_population(count, possible_values):
    """Create a population of random networks.
        """
    pop = []
    for _ in range(0, count):
        chromosome=[]
        for i in range(horizon):
            chromosome.append(random.choice(possible_values[i]))
        pop.append(np.array(chromosome,dtype=float))
    return pop

def breed(mother, father):
    """Make two children as parts of their parents.
    Args:
        mother (list)
        father (list)
    """
    children = []
    for _ in range(2):
        child = []
        # Loop through the parameters and pick params for the kid.
        for i in range(horizon):
            child.append(random.choice([mother[i], father[i]]))
        children.append(child)
    return children


def mutate(chromosome):
    """Randomly mutate one part of the network.
    Args:
        chromosome (list): The network parameters to mutate
    """
    # Choose a random index.
    mutation_index = random.choice(range(horizon))

    # Mutate one of the params.
    chromosome[mutation_index] = random.choice(possible_values[mutation_index])

    return chromosome

def predictions(index):
    state = scaler.transform(states[index])
    _, inversed_prediction = predict(state, forecast, model=models[index], horizon=horizon, scaler=scaler)
    return np.reshape(inversed_prediction, newshape=horizon)

def fitness(chromosome):
    global forecast
    forecast = np.concatenate((np.reshape(chromosome,newshape=[horizon,1]), np.reshape(temps,newshape=[horizon,1]), np.ones(shape=[horizon,1])), axis=1)
    forecast = scaler.transform(forecast)
    forecast = forecast[:,:-1]
    # R = 0
    # extra = 0
    # cost = 0
    # total_load = np.zeros(shape=horizon)
    preds = []
    for i in range(users_num):
        preds.append(predictions(i))

    total_load = sum(np.array(preds))
    R = np.linalg.multi_dot([np.array(chromosome),total_load])
    cost = np.linalg.multi_dot([np.array(pmin), total_load])+ sum(quadratic_price*np.power(total_load,2)) + fixed_cost

    if R > R_max:
        R -= R_penalty*(R-R_max)

    extra = np.sum((np.subtract(total_load, load_max) + np.abs(np.subtract(total_load, load_max))) / 2)
    print(extra)
    fitness = R-cost-overload_penalty*extra
    return fitness

def real_fitness(chromosome):

    total_load = np.zeros(shape=horizon)
    for index,params in enumerate(parameters.values):
        tipping = tippings(params[0], params[1], params[2])
        prediction, frames = get_real_sequence(states[index], chromosome, horizon, tipping)
        prediction = np.reshape(prediction,newshape=horizon)
        total_load = total_load + prediction
    R = np.linalg.multi_dot([np.array(chromosome),total_load])
    cost = np.linalg.multi_dot([np.array(pmin), total_load])+ sum(quadratic_price*np.power(total_load,2)) + fixed_cost
    if R > R_max:
        R -= R_penalty*(R-R_max)
    extra = np.sum((np.subtract(total_load, load_max) + np.abs(np.subtract(total_load, load_max))) / 2)
    # print(extra)
    fitness = R-cost-overload_penalty*extra
    print(fitness)
    return fitness


def evolve(pop, retain=.25, random_select=.20, mutate_chance=0.15):
    """Evolve a population of networks.
    Args:
        pop (list): A list of lists of prices
    """
    # Get scores for each chromosome.

    global origins_length
    global list_to_save

    #score only the generated children of the population
    graded = [np.append(np.array(fitness(chromosome)),chromosome) for chromosome in pop[origins_length:]]
    # graded = [np.append(np.array(real_fitness(chromosome)), chromosome) for chromosome in pop[origins_length:]]
    # get the parents from last population that we already know their fitness function
    graded.extend(list_to_save[:origins_length])
    # Sort on the scores.
    sorted_chromosomes = sorted(graded, key=lambda x: x[0],reverse=True)
    # graded = [x[-1] for x in sorted_chromosomes]

    #save for records
    list_to_save =  sorted_chromosomes

    # print(list_to_save)
    # Get the number we want to keep for the next gen.
    retain_length = int(len(sorted_chromosomes)*retain)

    # The parents are every network we want to keep.
    parents = np.array(sorted_chromosomes[:retain_length])[:,1:]

    # For those we aren't keeping, randomly keep some anyway.
    for individual in sorted_chromosomes[retain_length:]:
        if random_select > random.random():
            parents = np.append(parents, np.reshape(individual[1:],newshape=[1,horizon]),axis=0)
    origins_length = parents.shape[0]

    # Randomly mutate some of the networks we're keeping.
    for i, individual in enumerate(parents):
        if mutate_chance > random.random():
            individual = mutate(individual)
            parents[i] = individual

    # Now find out how many spots we have left to fill.
    parents_length = parents.shape[0]
    desired_length = len(pop) - parents_length
    children = []

    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:
        # Get a random mom and dad.
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)

        # Assuming they aren't the same network...
        if male != female:
            male = parents[male]
            female = parents[female]

            # Breed them.
            babies = breed(male, female)

            # Add the children one at a time.
            for baby in babies:
                # Don't grow larger than desired length.
                if len(children) < desired_length:
                    children.append(baby)

    parents = np.append(parents, np.reshape(np.array(children),newshape=[len(children),horizon]),axis=0)
    return parents, list_to_save

def to_csv(list_to_save, columns, header=True):
    results = pd.DataFrame(list_to_save, columns=columns)
    results.to_csv("GA_pricing1.csv", header=header, index=False)
    print(results)

def get_real_sequence(state,prices, prediction_len,tipping):
    forecast = np.concatenate((np.reshape(prices, newshape=[horizon, 1]), np.reshape(temps, newshape=[horizon, 1])),axis=1)
    real_frames = []
    curr_frame = state[-1]
    real = []
    for j in range(prediction_len):
        load = fuzzy(curr_frame[2], curr_frame[0], curr_frame[1],tipping=tipping)
        real.append(load)
        next_frame = np.append(forecast[j],real[-1])
        curr_frame = next_frame
        real_frames.append(curr_frame)
    return real, np.array(real_frames)

if __name__ == '__main__':
    global_start_time = time.time()
    # # epochs  = 10
    seq_len = 2
    prices_solutions = []
    loads_solutions = []
    states=[]
    for user in range(users_num):
        dataframe = pd.read_csv('fuzzy_out'+str(user)+'.csv')
        data = dataframe.values
        state = data[0:2, :]
        states.append(state)
    ori_loads = np.array(data[2:, 2], dtype=float)
    ori_prices = np.array(data[2:, 0], dtype=float)
    prices = ori_prices/1.5
    temperatures = np.array(data[2:, 1], dtype=float)

    global models
    models = []
    for i in range(users_num):
        models.append(lModel)

    # horizons = dataframe.shape[0] // horizon
    horizons = 1
    cols = list(range(horizon + 1))
    for h in range(horizons)[-1:]:
        temps = temperatures[h*horizon:(h+1)*horizon]
        pmin = prices[h*horizon:(h+1)*horizon]
        pmax = pmin * 2
        factor = np.average(ori_prices[h*horizon:(h+1)*horizon])*np.average(ori_loads[h*horizon:(h+1)*horizon])
        load_max = np.ones(shape=horizon) * np.average(ori_loads[h*horizon:(h+1)*horizon])*users_num
        R_max = horizon * factor * users_num
        possible_values = [list(frange(pmin[i], pmax[i], 0.01)) for i in range(horizon)]
        pop = create_population(100, possible_values)
        results_list = []
        origins_length = 0
        list_to_save = []
        maximums = []

        # Evolve
        for i in range(100):
            print('generation %d'%i)
            pop, list_to_save = evolve(pop)
            maximums.append(np.max(np.array(list_to_save)[:, 0]))
            # results_list.extend(list_to_save)
            # to_csv(list_to_save, header=False, columns=cols)

        # score only the generated children of the population
        # graded = [np.append(np.array(real_fitness(chromosome)), chromosome) for chromosome in pop[origins_length:]]
        graded = [np.append(np.array(fitness(chromosome)), chromosome) for chromosome in pop[origins_length:]]
        # get the parents from last population that we already know their fitness function
        graded.extend(list_to_save[:origins_length])
        # Sort on the scores.
        sorted_chromosomes = sorted(graded, key=lambda x: x[0], reverse=True)
        # save for records
        sorted_result_list = sorted_chromosomes
        # results_list.extend(list_to_save)
        # sorted_result_list = sorted(list_to_save, key=lambda x: x[0], reverse=True)
        # to_csv(sorted_result_list, columns=cols)

        best_solution = sorted_result_list[0]
        print('Best solution:')
        print(best_solution)
        prices_solutions.extend(list(best_solution[1:]))
        real_loads = []
        for user in range(users_num):
            params  = parameters.values[user, :]
            tipping = tippings(params[0], params[1], params[2])
            best_solution_real, frames = get_real_sequence(states[user],best_solution[1:], horizon, tipping)
            real_loads.append(best_solution_real)
            states.append(frames[-2:,:])
        states=states[users_num:]
        loads_solutions.append(real_loads)

        pyplot.plot(maximums)
        pyplot.xlabel("Iterations")
        pyplot.ylabel("Best fitness")
        pyplot.title("Learning process")
        pyplot.legend()
        pyplot.show()
        #
        # pyplot.plot(best_solution[1:], label = "best solution prices")
        # pyplot.plot(pmin, label="pmin")
        # pyplot.plot(pmax, label="pmax")
        # pyplot.legend()
        # pyplot.show()
        # # pyplot.plot(best_solution_predictions, label="best solution predictions")
        # pyplot.plot(best_solution_real, label="best solution real")
        # pyplot.plot(load_max, label="max load")
        # pyplot.legend()
        # pyplot.show()
    loads_solutions = np.array(loads_solutions)
    loads_solutions = np.reshape(np.transpose(loads_solutions,axes=[0,2,1]),newshape=[loads_solutions.shape[0]*loads_solutions.shape[2],loads_solutions.shape[1]])
    prices_solutions = np.reshape(np.array(prices_solutions),newshape=[len(prices_solutions),1])
    results = pd.DataFrame(data=np.concatenate((prices_solutions,loads_solutions),axis=1))
    results.to_csv("ref_optimized_prices_loads.csv", index=False)
