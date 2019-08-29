# Dependacies
# Install numpy  for python 3.6
# Install Skfuzzy for python 3.6

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

def tippings():
    # Declaring Values L np arange ((min value , max value , steps) , 'Name')
    temperature = ctrl.Antecedent(np.arange(-25, 36, 1), 'Temperature')
    price = ctrl.Antecedent(np.arange(0, 27,.1), 'Price')
    load = ctrl.Consequent(np.arange(0, 3.5, .1), 'Load')
    load1 = ctrl.Antecedent(np.arange(0, 3.5, .1), 'Load1')
    # Defining States [start , mid/ max , stop]
    load['low'] = fuzz.trimf(load.universe, [0, 0, 1.0])
    load['average'] = fuzz.trimf(load.universe, [0, 1.5, 2.0])
    load['high'] = fuzz.trimf(load.universe, [1.5, 3.5, 3.5])
    # You can see how these look with .view()
    # load.view()

    temperature['cold'] = fuzz.trimf(temperature.universe, [-25, -25, 20])
    temperature['average'] = fuzz.trimf(temperature.universe, [0, 20, 35])
    temperature['hot'] = fuzz.trimf(temperature.universe, [20, 35, 35])
    # temperature.view()

    price['low'] = fuzz.trimf(price.universe, [0, 0, 3])
    price['average'] = fuzz.trimf(price.universe, [0, 3, 6])
    price['high'] = fuzz.trimf(price.universe, [4, 26, 26])
    # price.view()

    # Defining Rules
    rule1 = ctrl.Rule(price['low'] & temperature['cold'], load['high'])
    rule9 = ctrl.Rule(price['low'] & temperature['average'], load['average'])
    rule6 = ctrl.Rule(price['low'] & temperature['hot'], load['high'])

    rule2 = ctrl.Rule(price['average'] & temperature['cold'], load['high'])
    rule4 = ctrl.Rule(price['average'] & temperature['average'], load['low'])
    rule7 = ctrl.Rule(price['average'] & temperature['hot'], load['high'])

    rule3 = ctrl.Rule(price['high'] & temperature['cold'], load['average'])
    rule5 = ctrl.Rule(price['high'] & temperature['average'], load['low'])
    rule8 = ctrl.Rule(price['high'] & temperature['hot'], load['average'])

    delta_Temperature= ctrl.Consequent(np.arange(-20, 20, 1), 'Delta_Temperature')

    delta_Temperature['negative']=fuzz.trimf(delta_Temperature.universe,[-20,-20,0])
    delta_Temperature['neutral']=fuzz.trimf(delta_Temperature.universe,[-20,0,20])
    delta_Temperature['positive']=fuzz.trimf(delta_Temperature.universe,[0,20,20])

    load1['low'] = fuzz.trimf(load1.universe, [0, 0, 1.0])
    load1['average'] = fuzz.trimf(load1.universe, [0, 1.5, 2.0])
    load1['high'] = fuzz.trimf(load1.universe, [1.5, 3.5, 3.5])

    # Defining Rules
    rule10 = ctrl.Rule(load1['low'], delta_Temperature['neutral'])
    # rule11 = ctrl.Rule(load1['low'] , delta_Temperature['neutral'])
    # rule12 = ctrl.Rule(load1['low'] & temperature['hot'], delta_Temperature['neutral'])
    rule13 = ctrl.Rule(load1['average'] & temperature['cold'], delta_Temperature['negative'])
    rule14 = ctrl.Rule(temperature['average'], delta_Temperature['neutral'])
    rule15 = ctrl.Rule(load1['average'] & temperature['hot'], delta_Temperature['neutral'])
    rule16 = ctrl.Rule(load1['high'] & temperature['hot'], delta_Temperature['positive'])
    # rule17 = ctrl.Rule(load1['high'] & temperature['average'], delta_Temperature['neutral'])
    rule18 = ctrl.Rule(load1['high'] & temperature['cold'], delta_Temperature['negative'])

    # Applying Rules
    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    tipping1 = ctrl.ControlSystemSimulation(tipping_ctrl)

    tipping_ctr2 = ctrl.ControlSystem([rule10, rule13, rule14, rule15, rule16,  rule18])
    tipping2 = ctrl.ControlSystemSimulation(tipping_ctr2)
    return tipping1, tipping2
# Getting Input
# open File and get Data
folder_path = 'PriceTemp.csv'
fout = open('fuzzy_out.csv', 'w')

num_users = 30
dataframe = pd.read_csv(folder_path).dropna()
data = dataframe.values

for i in range(num_users):
    price_value = 0
    temp_value = 0
    load_value = 0
    for line in open(folder_path):
        try:
            price_n_temp = line.split(',')
            price_value = float(price_n_temp[0])
            temp_value = float(price_n_temp[1].replace('\n',''))

        # print(str(price_value) + '.' + str(temp_value))
            print("price: " + str(price_value))
            print("Load: " + str(load_value))
            tipping2.input['Load1'] = load_value
            tipping2.input['Temperature'] = temp_value
            tipping2.compute()
            print("outdoor temperature: " + str(temp_value))
            indoor_temp=temp_value-tipping2.output['Delta_Temperature']
            print("indoor temperature: " + str(indoor_temp))
            tipping1.input['Price'] = price_value
            tipping1.input['Temperature'] = temp_value
            tipping1.compute()
            load_value=tipping1.output['Load']
        except:
            print("skipped")
            continue
        # Crunch the numbers

        fout.write('\n' + str(price_value) + ',' + str(temp_value) + ',' + str(tipping1.output['Load']))
        # print("WRITING")
    fout.close()