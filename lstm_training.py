import lstm
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pandas as pd
from network import Network
import random


# This script is for training an LSTM network.
# The parameters of the network are tuned using a genetic algorithm

# Global variables

nn_param_choices={'seq_len': range(1,6,1),
                  'lstm_size':range(10,50,10),
                    'num_lstm':range(1,5),
                    'dropout': [x*0.1 for x in range(0,4,1)],
                  'activation': ['relu', 'elu','selu','tanh','sigmoid','hard_sigmoid'],
                  'recurrent_activation':['relu', 'elu','selu','tanh','sigmoid','hard_sigmoid'],
                  'optimizer': ['adadelta','adagrad','rmsprop', 'adam','adamax','nadam','sgd']
        }


## This scripts is used to build the model and train it used training data.
# Main Run Thread
def train_score(network={}):
    print(network)
    print('> Loading data... ')
    # global seq_len
    # global num_features
    X_train, y_train, X_test, y_test, scaler = lstm.load_data('fuzzy_out0.csv', network['seq_len'])
    num_features = X_train.shape[2]
    dataset = [X_train, y_train, X_test, y_test]
    # initialize model according to the given values of the network
    model = lstm.build_model(input_shape=[network['seq_len'], num_features],
                             lstm_size=network['lstm_size'],
                             num_lstm=network['num_lstm'],
                             dropout=network['dropout'],
                             activation=network['activation'],
                             recurrent_activation=network['recurrent_activation'],
                             optimizer=network['optimizer'])
    model.fit(
        dataset[0],
        dataset[1],
        validation_split=0.2)
    loss = model.evaluate(x=dataset[2], y=dataset[3])

    print('Training duration (s) : ', time.time() - global_start_time)
    # model.save('model.h5')
    # predictions = lstm.predict(model, dataset[2])
    # # global scaler
    # try:
    #     predicted_load = lstm.inverse_transform(dataset[2], predictions, scaler)
    #     true_load = lstm.inverse_transform(dataset[2], dataset[3], scaler)
    #
    #     rmse = sqrt(mean_squared_error(true_load, predicted_load))
    #     mape = np.mean(np.abs((true_load - predicted_load) / true_load)) * 100
    # except Exception as e:
    #     print(e)
    #     rmse=100.0
    #     mape=100.0
    # print('Test RMSE: %.3f' % rmse)
    # print('Test MAPE: %.3f ' % mape)
    #
    # # pyplot.plot(true_load, label='True')
    # # pyplot.plot(predicted_load,'--', label='predicted')
    # # pyplot.legend()
    # # pyplot.show()
    return loss

def create_population(count):
    """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of network objects
        """
    pop = []
    for _ in range(0, count):
        # Create a random network.
        network = Network(nn_param_choices)
        network.create_random()

        # Add the network to our population.
        pop.append(network)
    return pop

def breed(mother, father):
    """Make two children as parts of their parents.
    Args:
        mother (dict): Network parameters
        father (dict): Network parameters
    """
    children = []
    for _ in range(2):

        child = {}

        # Loop through the parameters and pick params for the kid.
        for param in nn_param_choices:
            child[param] = random.choice(
                [mother.network[param], father.network[param]]
            )

        # Now create a network object.
        network = Network(nn_param_choices)
        network.create_set(child)

        children.append(network)

    return children


def mutate(network):
    """Randomly mutate one part of the network.
    Args:
        network (dict): The network parameters to mutate
    """
    # Choose a random key.
    mutation = random.choice(list(nn_param_choices.keys()))

    # Mutate one of the params.
    network.network[mutation] = random.choice(nn_param_choices[mutation])

    return network


def evolve(pop, retain=.25, random_select=.20, mutate_chance=0.15):
    """Evolve a population of networks.
    Args:
        pop (list): A list of network parameters
    """
    # Get scores for each network.

    global origins_length
    global list_to_save

    #score only the generated children of the population
    graded = [[network, train_score(network.network)] for network in pop[origins_length:]]

    #get the parents from last population that we already know their fitness function
    graded.extend([[pop[i], list_to_save[i][-1]] for i in range(origins_length)])
    # Sort on the scores.
    sorted_nets = sorted(graded, key=lambda x: x[-1])
    graded = [x[0] for x in sorted_nets]

    #save for records
    list_to_save =  [list(element[0].network.values())+[element[-1]] for element in sorted_nets ]

    print(list_to_save)

    # Get the number we want to keep for the next gen.
    retain_length = int(len(graded)*retain)

    # The parents are every network we want to keep.
    parents = graded[:retain_length]

    # For those we aren't keeping, randomly keep some anyway.
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)


    origins_length = len(parents)

    # Randomly mutate some of the networks we're keeping.
    for individual in parents:
        if mutate_chance > random.random():
            individual = mutate(individual)

    # Now find out how many spots we have left to fill.
    parents_length = len(parents)
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

    parents.extend(children)
    return parents, list_to_save

def to_csv(list_to_save, columns, header=True ):
    results = pd.DataFrame(list_to_save, columns=columns)
    with open('results.csv', 'a') as f:
        results.to_csv(f, header=header)
    print(results)

if __name__ == '__main__':
    global_start_time = time.time()
    # # epochs  = 10
    # seq_len = 2

    pop = create_population(50)

    print('> Data Loaded. Compiling...')

    results_list = []
    origins_length = 0
    list_to_save = []
    cols = list(nn_param_choices.keys())
    cols.extend(["loss"])
    to_csv(list_to_save,columns=cols)


    for i in range(50):
        print('generation %d'%i)
        pop, list_to_save = evolve(pop)
        results_list.extend(list_to_save)
        to_csv(list_to_save, header=False, columns=cols)


    #score only the generated children of the population
    graded = [list(train_score(network.network))+[network] for network in pop[origins_length:]]

    #get the parents from last population that we already know their fitness function
    graded.extend([[pop[i], list_to_save[i][-1]] for i in range(origins_length)])
    # Sort on the scores.
    sorted_nets = sorted(graded, key=lambda x: x[-1])
    list_to_save = [list(element[0].network.values()) + [element[-1]] for element in sorted_nets]
    results_list.extend(list_to_save)
    sorted_result_list = sorted(results_list, key=lambda x: x[-1])
    print(sorted_result_list)

    to_csv(list_to_save, header=False, columns=cols)


    # pyplot.plot(true_load, label='True')
    # pyplot.plot(predicted_load, label='predicted')
    # pyplot.legend()
    # pyplot.show()


