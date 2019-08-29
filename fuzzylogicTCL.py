# Dependacies
# Install numpy  for python 3.6
# Install Skfuzzy for python 3.6

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import random
import sys

sys.path.insert(0, 'C:/Users/tahanak/PycharmProjects/TCL/Taha_code')
from Simulation import createTCLs


# from TCL import TCL


def tippings(alpha, beta, gamma):
    # Declaring Values L np arange ((min value , max value , steps) , 'Name')
    temperature = ctrl.Antecedent(np.arange(-25, 36, 0.5), 'Temperature')
    price = ctrl.Antecedent(np.arange(0, 40.5, 0.05), 'Price')
    load = ctrl.Consequent(np.arange(0, 3.5, .01), 'Load')
    load1 = ctrl.Antecedent(np.arange(0, 3.5, .01), 'Load1')
    # Defining States [start , mid/ max , stop]
    load['much_low'] = fuzz.trimf(load.universe, [0.0, 0.0, 0.05])
    load['low'] = fuzz.trimf(load.universe, [0.0, 0.05, 1.0+beta])
    load['average'] = fuzz.trimf(load.universe, [0.05, 1.0+beta, 1.5+beta])
    load['high'] = fuzz.trimf(load.universe, [1.0+beta, 1.5+beta, 2.5+beta])
    load['much_high'] = fuzz.trimf(load.universe, [1.5+beta, 3.5, 3.5])
    # You can see how these look with .view()
    # load.view()

    temperature['much_cold'] = fuzz.trimf(temperature.universe, [-25, -25, 5])
    temperature['cold'] = fuzz.trimf(temperature.universe, [-25, 5, 20])
    temperature['average'] = fuzz.trimf(temperature.universe, [5, 20, 25])
    temperature['hot'] = fuzz.trimf(temperature.universe, [20, 36, 36])
    # temperature.view()

    price['much_low'] = fuzz.trimf(price.universe, [0, 0, 2.4+alpha])
    price['low'] = fuzz.trimf(price.universe, [0, 2.4+alpha, 3.0+alpha])
    price['average'] = fuzz.trimf(price.universe, [2.4+alpha, 3.0+alpha, 3.75+alpha])
    price['high'] = fuzz.trimf(price.universe, [3.0+alpha, 3.75+alpha, 6.0+alpha])
    price['much_high'] = fuzz.trimf(price.universe, [3.75+alpha, 40.5, 40.5])
    # price.view()

    # Defining Rules
    rule1 = ctrl.Rule(price['much_low'] & temperature['much_cold'], load['much_high'])
    rule2 = ctrl.Rule(price['much_low'] & temperature['cold'], load['high'])
    rule3 = ctrl.Rule(price['much_low'] & temperature['average'], load['average'])

    rule4 = ctrl.Rule(temperature['hot'], load['much_low'])

    rule5 = ctrl.Rule(price['low'] & temperature['much_cold'], load['much_high'])
    rule6 = ctrl.Rule(price['low'] & temperature['cold'], load['high'])
    rule7 = ctrl.Rule(price['low'] & temperature['average'], load['low'])

    rule8 = ctrl.Rule(price['average'] & temperature['much_cold'], load['much_high'])
    rule9 = ctrl.Rule(price['average'] & temperature['cold'], load['high'])
    rule10 = ctrl.Rule(price['average'] & temperature['average'], load['much_low'])

    rule11 = ctrl.Rule(price['high'] & temperature['much_cold'], load['high'])
    rule12 = ctrl.Rule(price['high'] & temperature['cold'], load['average'])
    rule13 = ctrl.Rule(price['high'] & temperature['average'], load['much_low'])

    rule14 = ctrl.Rule(price['much_high'] & temperature['much_cold'], load['average'])
    rule15 = ctrl.Rule(price['much_high'] & temperature['cold'], load['low'])
    rule16 = ctrl.Rule(price['much_high'] & temperature['average'], load['much_low'])

    delta_Temperature = ctrl.Consequent(np.arange(-20, 20, 1), 'Delta_Temperature')

    delta_Temperature['negative'] = fuzz.trimf(delta_Temperature.universe, [-20, -20, 0+gamma])
    delta_Temperature['neutral'] = fuzz.trimf(delta_Temperature.universe, [-20, 0+gamma, 20])
    delta_Temperature['positive'] = fuzz.trimf(delta_Temperature.universe, [0+gamma, 20, 20])


    load1['low'] = fuzz.trimf(load1.universe, [0.0, 0.0, 1.0+beta])
    load1['average'] = fuzz.trimf(load1.universe, [0.0, 1.0+beta, 1.5+beta])
    load1['high'] = fuzz.trimf(load1.universe, [1.0+beta, 3.5, 3.5])


    # Defining Rules
    rule17 = ctrl.Rule(load1['low'], delta_Temperature['neutral'])
    rule18 = ctrl.Rule(load1['average'] & temperature['cold'], delta_Temperature['negative'])
    rule19 = ctrl.Rule(temperature['average'], delta_Temperature['neutral'])
    rule20 = ctrl.Rule(load1['average'] & temperature['hot'], delta_Temperature['neutral'])
    rule21 = ctrl.Rule(load1['high'] & temperature['hot'], delta_Temperature['positive'])
    rule22 = ctrl.Rule(load1['high'] & temperature['cold'], delta_Temperature['negative'])

    # Applying Rules
    tipping_ctrl = ctrl.ControlSystem(
        [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule16, rule14,
         rule15])
    tipping1 = ctrl.ControlSystemSimulation(tipping_ctrl)

    tipping_ctr2 = ctrl.ControlSystem([rule17, rule18, rule19, rule20, rule21, rule22])
    tipping2 = ctrl.ControlSystemSimulation(tipping_ctr2)

    return tipping1, tipping2


# tipping = tippings()


def fuzzy(load_value, price_value, temp_value, tipping):
    # print('outdoor temperature: ' + str(temp_value))
    tipping[1].input['Load1'] = load_value
    tipping[1].input['Temperature'] = temp_value
    tipping[1].compute()
    detlta_temp = tipping[1].output['Delta_Temperature']
    temp_value -=  detlta_temp
    # print('indoor temperature: ' + str(temp_value))
    tipping[0].input['Price'] = price_value
    tipping[0].input['Temperature'] = temp_value
    tipping[0].compute()
    return tipping[0].output['Load']


if __name__ == '__main__':
    # Getting Input
    # open File and get Data
    folder_path = 'PriceTemp.csv'
    num_users = 30
    parameters = []
    dataframe = pd.read_csv(folder_path).dropna()
    data = dataframe.values
    for i in range(num_users):
        alpha = random.uniform(1.0,.7)
        beta = random.uniform(0,.4)
        gamma = random.uniform(1,3)
        tipping = tippings(alpha, beta, gamma)
        parameters.append([alpha,beta, gamma])
        price_value = 0
        temp_value = 0
        load_value = 0
        print('user: ' + str(i))
        fout = open('fuzzy_out' + str(i) + '.csv', 'w')
        fout.write('Price,Outdoor Temperature,Load')
        for t, line in enumerate(data[0:-1]):
            try:
                price_value = data[t + 1, 0] * 1.5
                # print("price value: " + str(price_value))
                temp_value = line[1]
                load_value = fuzzy(load_value,price_value, temp_value, tipping=tipping)
                # print("load value: " + str(load_value))
                # print('------------------------------------')
            except ValueError as e:
                print('Step skipped'+str(e))
                continue
            # Crunch the numbers

            fout.write(
                '\n' + str(price_value) + ',' + str(temp_value) + ',' + str(load_value) )
        # print("WRITING")
        fout.close()
    np.savetxt("parameters.csv", np.array(parameters), delimiter=",")
