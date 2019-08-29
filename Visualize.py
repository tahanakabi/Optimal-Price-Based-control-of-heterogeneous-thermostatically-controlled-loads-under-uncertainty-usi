import lstm
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import pandas as pd
import pickle
from lstm_test import predict, positive_values
from scipy import stats
from GA_pricing import fixed_cost, quadratic_price, R_penalty, overload_penalty, get_real_sequence

seq_len = 2
horizon = 24
num_users = 30
# dataframe = pd.read_csv('fuzzy_out0.csv')
def predictions_to_csv():
    for i in range(num_users):
        model = load_model('lstm_model'+str(i)+'.h5')
        X_train, y_train, X_test, y_test, scaler = lstm.load_data('fuzzy_out'+str(i)+'.csv', seq_len, validation_percentage=0)
        predictions = positive_values(lstm.predict(model, X_train))
        norm_data = X_train[:,0,:]
        data = scaler.inverse_transform(norm_data)
        norm_data = np.append(norm_data[:,:-1],predictions,axis=1)
        new_data = scaler.inverse_transform(norm_data)
        data = np.append(new_data,np.reshape(data[:,-1],[data.shape[0],1]),axis=1)
        np.savetxt("user"+str(i)+".csv", np.array(data), delimiter=",")
        print('saved as :'+"user"+str(i)+".csv")

def plot_by_user():
    for i in range(0,num_users,10):
        dataframe= pd.read_csv("user"+str(i)+".csv",header=None)
        true_loads = dataframe.values[:,2]
        predictions = dataframe.values[:,-1]
        prices = dataframe.values[:,0]
        temperatures = dataframe.values[:,1]
        step = true_loads.shape[0] // 10
        t = np.arange(0,24)
        # for k in range(10):
        #     index = k*step
        #     plt.plot(t,true_loads[index:index+24],'-', label= "True")
        #     plt.plot(t,predictions[index:index+24], '--', label="Predicted")
        #     plt.xlabel('time [h]')
        #     plt.ylabel('Power [kW]')
        #     plt.title('Day ' + str(index // 24+1) +" - user"+str(i+1))
        #     plt.legend()
        #     # plt.show()
        #     plt.savefig('Day' + str(index // 24+1)+"user"+str(i+1))
        #     plt.cla()

    for i in range(10):
        index = i * step
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time (h)')
        ax1.set_ylabel('Temperature [°C]', color='tab:red')
        ax1.plot(t, temperatures[index:index + 24], 'r-', label = "Temperature")
        ax1.tick_params(axis='y', labelcolor = 'tab:red')
        plt.legend(loc = 'upper right')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Price [cents/kWh]', color='tab:blue')
        ax2.plot(t, prices[index:index + 24], "b--", label = "Price")
        ax2.tick_params(axis='y', labelcolor = 'tab:blue')
        fig.tight_layout()
        plt.legend(loc = 'upper left')
        # otherwise the right y-label is slightly clipped
        plt.title('Day ' + str(index // 24+1))
        # plt.show()
        plt.savefig('Day' + str(index // 24+1))
        plt.cla()

if __name__== "__main__":
    h=24
    all_loads = []
    day = 0
    for user in range(num_users):
        dataframe = pd.read_csv('fuzzy_out' + str(user) + '.csv')
        data = dataframe.values
        loads = np.array(data[h*day+2:, 2], dtype=float)
        all_loads.append(loads)

    temperatures = np.array(data[h*day+2:, 0], dtype=float)
    prices = np.array(data[h*day+2:, 0], dtype=float)
    market_prices = prices/1.5
    all_loads = np.transpose(np.array(all_loads),axes=[1,0])
    bills = np.linalg.multi_dot([prices[:24], all_loads[:24]])
    # original_data = pd.DataFrame(data = all_loads, index=None)
    # print(original_data.head())
    total_loads = np.sum(all_loads, axis=1)
    revenue = np.multiply(prices, total_loads)
    penalty = 0
    # if sum(revenue[:24]) > R_max:
    #     penalty = R_penalty*(sum(revenue)-R_max)
    costs =  np.multiply(market_prices, total_loads) + quadratic_price*np.power(total_loads,2)+fixed_cost*np.ones(shape=market_prices.shape[0])
    # extra = (total_loads[:24] - load_max + np.abs(total_loads[:24] - load_max)) / 2
    extra = 0
    profit = revenue[:24]-costs[:24]-overload_penalty*extra
    total_revenue = sum(revenue[:24])
    total_profit = sum(profit)- penalty
    print(total_profit)
####################################################################################################################################################
    results_data = pd.read_csv('optimized_prices_loads.csv').values
    optimized_prices = np.array(results_data[:, 0], dtype=float)
    # optimized_prices = np.array(market_prices[:24]*2, dtype=float)
    all_loads_optimized = np.array(results_data[:, 1:], dtype=float)
    optimized_bills = np.linalg.multi_dot([optimized_prices, all_loads_optimized])
    optimized_total_loads = np.sum(all_loads_optimized,axis=1)
    optimized_revenue = np.multiply(optimized_prices, optimized_total_loads)
    penalty = 0
    # if sum(optimized_revenue) > R_max:
    #     penalty = R_penalty * (sum(optimized_revenue) - R_max)
    optimized_costs = np.multiply(market_prices[:24], optimized_total_loads)+quadratic_price*np.power(optimized_total_loads,2)+fixed_cost*np.ones(shape=optimized_total_loads.shape[0])
    # optimized_extra = (np.subtract(optimized_total_loads, load_max) + np.abs(np.subtract(optimized_total_loads, load_max))) / 2
    optimized_extra = 0
    optimized_profit = optimized_revenue - optimized_costs - overload_penalty*optimized_extra
    total_optimized_revenue = sum(optimized_revenue)
    optimized_total_profit = sum(optimized_profit) - penalty
    print(optimized_total_profit)
########################################################################################################################################################
    ref_results_data = pd.read_csv('ref_optimized_prices_loads.csv').values
    ref_optimized_prices = np.array(ref_results_data[:, 0], dtype=float)
    # optimized_prices = np.array(market_prices[:24]*2, dtype=float)
    ref_all_loads_optimized = np.array(ref_results_data[:, 1:], dtype=float)
    ref_optimized_bills = np.linalg.multi_dot([ref_optimized_prices, ref_all_loads_optimized])
    ref_optimized_total_loads = np.sum(ref_all_loads_optimized, axis=1)
    ref_optimized_revenue = np.multiply(ref_optimized_prices, ref_optimized_total_loads)
    ref_penalty = 0
    # if sum(optimized_revenue) > R_max:
    #     penalty = R_penalty * (sum(optimized_revenue) - R_max)
    ref_optimized_costs = np.multiply(market_prices[:24], ref_optimized_total_loads) + quadratic_price * np.power(
        ref_optimized_total_loads, 2) + fixed_cost * np.ones(shape=ref_optimized_total_loads.shape[0])
    # optimized_extra = (np.subtract(optimized_total_loads, load_max) + np.abs(np.subtract(optimized_total_loads, load_max))) / 2
    ref_optimized_extra = 0
    ref_optimized_profit = ref_optimized_revenue - ref_optimized_costs - overload_penalty * ref_optimized_extra
    total_ref_revenue = sum(ref_optimized_revenue)
    ref_optimized_total_profit = sum(ref_optimized_profit) - ref_penalty
    print(ref_optimized_total_profit)
#########################################################################################################################################################

    index = np.arange(num_users)
    bar_width = 0.1
    # plt.bar(index , ref_optimized_bills, bar_width,
    #         color='g',
    #         label='Bills Under Benchmark Optimal Prices ')
    # plt.bar(index + bar_width, optimized_bills, bar_width,
    #         color='b',
    #         label='Bills Under Optimized Prices')
    #
    # plt.xlabel('Users')
    # plt.ylabel('Daily Bill (€ cents)')
    # plt.title('Daily Bills by User')
    # plt.xticks(index + bar_width, list(range(num_users+1))[1:])
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('Daily Bills by User')

    originals=[total_revenue,total_profit]
    optimized = [total_optimized_revenue, optimized_total_profit]
    reference = [total_ref_revenue, ref_optimized_total_profit]
    index=np.arange(2)
    tick=["Revenue","Profit"]
    plt.bar(index,originals , bar_width,
            color='g',
            label='Under Original Prices ')
    plt.bar(index+ bar_width, optimized, bar_width,
            color='b',
            label='Under Optimized Prices ')
    plt.bar(index+ 2*bar_width, reference, bar_width,
            color='r',
            label='Under Benchmark Prices ')

    plt.ylabel('€ cents')
    plt.title('Total Revenues and Profits')
    plt.xticks(index + bar_width, tick)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plt.plot(ref_optimized_total_loads[:24], '-',label='Benchmark Loads')
    # plt.plot(optimized_total_loads, '--', label='Optimized Loads')
    # plt.xlabel('Time (h)')
    # plt.ylabel('Loads (kWh)')
    # plt.title('Electricity consumption')
    # plt.legend()
    # plt.show()
    # # plt.savefig('Total Electricity consumption')
    #
    # plt.plot(profit[:24], 'o--', color='r', label = 'Profit Under Original Prices ')
    # plt.plot(optimized_profit[:24], '+--', color='b', label = 'Profit Under Optimized Prices')
    # plt.plot(revenue[:24], 'o--', color='k', label='Revenue Under Original Prices ')
    # plt.plot(optimized_revenue[:24], '+--', color='g', label='Revenue Under Optimized Prices')
    # plt.legend('Revenue and Profit')
    # plt.xlabel('Time (h)')
    # plt.ylabel('€ cents')
    # plt.title('Revenues And Profits')
    # plt.legend()
    # plt.show()
    # plt.savefig('Revenue and Profit')

    # plt.plot(ref_optimized_prices[:24], 'b-', label = 'Benchmark Prices')
    # plt.plot(optimized_prices[:24], '--', label = 'Optimized Prices')
    # plt.plot(market_prices[:24], 'g-', label='Minimum Prices')
    # plt.plot(market_prices[:24]*2, 'r-', label='Maximum Prices')
    # plt.xlabel('Time (h)')
    # plt.ylabel('Electricity Prices (cents/kWh)')
    # plt.title('Electricity Prices')
    # plt.legend()
    # plt.show()
    # plt.savefig('Electricity Prices')





    # true_loads = []
    # predictions = []
    #
    #
    # plot_by_user()
    #
    #
    #
    # for i in range(0,num_users,1):
    #     dataframe= pd.read_csv("user"+str(i)+".csv",header=None)
    #     true_loads.append(dataframe.values[:,2])
    #     predictions.append(dataframe.values[:,-1])
    #     step = true_loads[0].shape[0] // 10
    #     t = np.arange(0,24)
    #
    #
    # for k in range(10):
    #     index = k*step
    #     tr_ld = np.array(true_loads)[:,index:index+24]
    #     mean_true_load = np.mean(tr_ld, axis=0)
    #     se_true_load = stats.sem(tr_ld, axis=0)
    #     sum_true_load = np.sum(tr_ld, axis=0)
    #     plt.plot(t,mean_true_load,'k-', label= "True")
    #     plt.fill_between(t, mean_true_load - 1.96*se_true_load, mean_true_load + 1.96*se_true_load, color='silver')
    #     pr_ld = np.array(predictions)[:, index:index + 24]
    #     mean_predictions = np.mean(pr_ld, axis=0)
    #     se_predictions = stats.sem(pr_ld, axis=0)
    #     sum_predictions = np.sum(pr_ld, axis=0)
    #     plt.plot(t, mean_predictions, 'r-', label="Predicted")
    #     plt.fill_between(t, mean_predictions - 1.96 * se_predictions, mean_predictions + 1.96 * se_predictions, color='coral')
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Power [kW]')
    #     plt.title('Day ' + str(index // 24+1) )
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('Day' + str(index // 24+1)+"envelope")
    #     plt.cla()
    #
    #     plt.xlabel('time [h]')
    #     plt.ylabel('Power [kW]')
    #     plt.plot(t, sum_true_load, 'k-', label="True")
    #     plt.plot(t, sum_predictions, 'r-', label="Predicted")
    #     plt.title('Day ' + str(index // 24 + 1) )
    #     plt.legend()
    #     # plt.show()
    #     plt.savefig('Day' + str(index // 24 + 1) + "overall")
    #     plt.cla()

