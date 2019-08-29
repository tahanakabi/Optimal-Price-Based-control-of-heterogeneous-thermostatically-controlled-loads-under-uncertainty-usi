import pandas as pd



def get_temperatures(filename):
    data = []
    with open(filename, 'r') as d:  # using the with keyword ensures files are closed properly
        for line in d.readlines():
            parts = line.split(';')
            price = parts[5].replace(",", ".").replace("\n", "")
            try:
                tempF = float(price)
            except:
                tempF = value
            value=tempF
            data.append(value)
    return data

if __name__ == '__main__':
    temps = get_temperatures('temp.txt')
    df= pd.Series(temps,dtype=float)
    df.to_csv('temperatures.csv')

