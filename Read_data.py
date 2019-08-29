import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_X_y(filename,test_size):

    prices=[]
    loads=[]
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines()[1:]:
            parts = line.split(',')
            load = [float(parts[-1].replace(",", "."))]
            # if load in loads:
                # continue
            price_temp = [float(i.replace(",", ".").replace("\n", "")) for i in parts[0:2]]
            prices.append(price_temp)
            loads.append(load)

    X=np.array(prices)

    y=np.array(loads)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    return X_train, X_test, y_train, y_test


def get_results(filename):
    loads=[]
    predictions = []
    residuals = []
    with open(filename, 'r' ) as d:
        lines=d.readlines()
        for idx in range(len(lines))[1::24]:
            l=[]
            p=[]
            r=[]
            for ln in lines[idx:idx+24]:
                parts = ln.split(',')
                l.append(float(parts[1].replace(",", ".").replace("\n", "")))
                p.append(float(parts[2].replace(",", ".").replace("\n", "")))
                r.append(float(parts[3].replace(",", ".").replace("\n", "")))

            loads.append(l)
            predictions.append(p)
            residuals.append(r)

    return np.array(loads), np.array(predictions), np.array(residuals)



