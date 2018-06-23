#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:11:43 2018

@author: ali
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata


'''
create a return database
'''
def create_return_database(risk_free_rate=0.02):
    # stingy stock returns
    index_stingy_stocks = [x for x in range(2017, 2001, -1)]
    r_stingy_stocks = [58.3, 
                       8.4, 
                       -1.4, 
                       6.9, 
                       37.8, 
                       10.9, 
                       -16.1, 
                       69.4, 
                       64.5, 
                       -40.1, 
                       -5.5, 
                       28.9, 
                       29.2,
                       29.8, 
                       33.8, 
                       -1.9]
    r_stingy_stocks = [x/100 for x in r_stingy_stocks]
    df_stingy_stocks = pd.DataFrame(data=r_stingy_stocks, 
                                    index=index_stingy_stocks,
                                    columns=['stingy'])
    
    # Defensive Graham Stocks
    index_defensive_graham_stocks = [x for x in range(2017, 2000, -1)]
    r_defensive_graham_stocks = [12.8, 
                                 1.6,
                                 2.2,
                                 5.1,
                                 19.0,
                                 26.6,
                                 4.1,
                                 2.3,
                                 2.2,
                                 -6.5,
                                 34.4,
                                 -3.8,
                                 46.6,
                                 32.2,
                                 56.8,
                                 28.2,
                                 20.2]
    r_defensive_graham_stocks = [x/100 for x in r_defensive_graham_stocks]
    df_defensive_graham_stocks = pd.DataFrame(data=r_defensive_graham_stocks, 
                                    index=index_defensive_graham_stocks,
                                    columns=['graham'])
    
    # MoneySense 200
    index_money_sense_200 = [x for x in range(2017, 2004, -1)]
    r_money_sense_200 = [13.5,
                         2.8,
                         2.1,
                         2.3,
                         55.0,
                         13.4,
                         -4.2,
                         19.7,
                         41.0,
                         -32.9,
                         16.2,
                         37.6,
                         57.6]
    r_money_sense_200 = [x/100 for x in r_money_sense_200]
    df_money_sense_200 = pd.DataFrame(data=r_money_sense_200,
                                      index=index_money_sense_200,
                                      columns=['money_sense'])
    
    # cash returns
    index_cash = index_defensive_graham_stocks.copy()
    r_cash = [risk_free_rate for x in index_cash]
    df_cash = pd.DataFrame(data=r_cash, index=index_cash, columns=['cash'])
    
    
    df_combined = df_defensive_graham_stocks.join(df_stingy_stocks)
    df_combined = df_combined.join(df_money_sense_200)
    df_combined = df_combined.join(df_cash)
    df_combined = df_combined.reindex(index=df_combined.index[::-1])
    
    return df_combined


'''
calculate the sortino ratio
'''
def sortino(df, threshold=0.02):
    downside_std = df.clip_upper(threshold).std(ddof=1)
    return (df.mean()-threshold)/downside_std


'''
calculate the sharpe ratio
'''
def sharpe(df, risk_free=0.02):
    return (df.mean()-risk_free)/df.std(ddof=1)

'''
simulate a portfolio of stocks with allowance for leverage
'''
def simulate_portfolio(df_returns, 
                       allocations={'stingy':1.0,
                                    'graham':4.5,
                                    'money_sense': 0.5, 
                                    'cash':-5.0}, 
                       initial_investment=10000, 
                       annual_adds=10000,
                       risk_free_rate=0.02, 
                       borrowed_interest=0.0365,
                       min_cash=-100000):
    
    # start of year value
    time = [x for x in df_returns.index]
    
    # portfolio returns
    portfolio_returns = []
    equity_start = []
    equity_end = []
    cash_history = []
    allocations_history = []
    current_return = 0
    
    # variables for calculating drawdown
    max_draw_down = 0
    high_water_mark = 0
    
    cash = 0
    
    for year in time:
        
        if year == time[0]:
            current_equity = initial_investment
        else:
            current_equity += annual_adds
        equity_start += [current_equity]
        high_water_mark = max(high_water_mark, current_equity)
        
        # adjust allocations based on availability of portfolios and borrowed cash
        current_allocations = {}
        for item in allocations:
            if not df_returns.isnull()[item][year]:
                current_allocations[item] = allocations[item]
        # adjust cash value and allocation 
        cash = min(max(current_allocations['cash'] * current_equity, min_cash), cash)
        cash_history += [cash]
        # adjust/normalize allocations 
        current_allocations['cash'] *= cash/(current_allocations['cash'] * current_equity)
        sum_portfolio_allocations = sum(current_allocations.values()) - current_allocations['cash']
        for item in current_allocations:
            if item is not 'cash':
                current_allocations[item] *= (1-current_allocations['cash'])/sum_portfolio_allocations
        allocations_history += [current_allocations]
        
        # calculate current return
        current_return = 0
        for item in current_allocations:
            if item is not 'cash':
                current_return += df_returns[item][year] * current_allocations[item]
            else:
                if cash >= 0:
                    current_return += df_returns['cash'][year] * current_allocations[item]
                else:
                    current_return += borrowed_interest * current_allocations[item]
        portfolio_returns += [current_return]
        
        # calculate current equity, etc. based on returns
        current_equity += current_return * current_equity
        equity_end += [current_equity]
        max_draw_down = max(max_draw_down, high_water_mark-current_equity)
    
    result_df = pd.DataFrame(data={'equity_start': equity_start,
                                   'equity_end': equity_end,
                                   'return': portfolio_returns,
                                   'cash': cash_history,
                                   'allocations': allocations_history}, index=time)
    result_df = result_df.reindex_axis(['equity_start', 'equity_end', 'return', 
                                        'cash', 'allocations'], axis=1)
    
    sharpe_ratio = sharpe(result_df['return'], risk_free_rate)
    sortino_ratio = sortino(result_df['return'], risk_free_rate)
    
    results_dict = {'results_df': result_df,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_draw_down': max_draw_down,
                    'target_allocation': allocations,
                    'min_cash': min_cash}
    
    return results_dict
  
    
'''
plot portfolio components over time
'''
def plot_portfolio(df_results, value=True):
    
    # prepare data
    x_values = df_results.index.values
    y_values = {}
    for item in x_values:
        for p in df_results['allocations'][item]:
            if p not in y_values:
                y_values[p] = []
            if value:
                y_values[p].append(df_results['allocations'][item][p] * df_results['equity_start'][item])
            else:
                y_values[p].append(df_results['allocations'][item][p])
    cash_values = y_values['cash']
    del y_values['cash']
    neg_cash_values = [-x for x in cash_values]
    # build y_values_list
    y_values_list = []
    y_values_list += [cash_values]
    y_values_list += [neg_cash_values]
    y_values_list += [y_values[x] for x in y_values]
    l_n = max([len(x) for x in y_values_list])
    y_values_updated_list = [[0]*(l_n-len(x))+x for x in y_values_list]
    y_stacked = np.vstack(y_values_updated_list)
    
    # build labels
    labels = ['', 'borrowed'] + [x for x in y_values]
    
    # plot data
    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.stackplot(x_values, y_stacked, labels=labels)
    ax.legend(loc=2)
    ax.set_xlabel('Year')
    if value:
        ax.set_ylabel('Value, $')
    else:
        ax.set_ylabel('Allocation')
    plt.show()
    
    return


allocations_g = []
'''
perofrm monte carlo simulation for portfolio allocations.
save each allocation in allocation_g list (global) for debugging purposes.
'''
def monte_carlo_allocations(df_returns, 
                            num_iterations=10000,
                            initial_investment=10000, 
                            annual_adds=10000,
                            risk_free_rate=0.02, 
                            borrowed_interest=0.0365,
                            cash_alloc=-0.3,
                            min_cash=-100000):
    
    global allocations_g
    results = []
    
    for i in tqdm(range(num_iterations)):
        # build allocations
        rand_1 = round(np.random.random(),2)
        rand_2 = round(np.random.random(),2)
        while rand_2==rand_1:
            rand_2 = round(np.random.random(),2)
        allocations = {}
        allocations['cash'] = cash_alloc
        allocations['stingy'] = round(min(rand_1, rand_2) * (1-cash_alloc), 2)
        allocations['graham'] = round((max(rand_1, rand_2) - min(rand_1, rand_2)) * (1-cash_alloc), 2)
        allocations['money_sense'] = 1.0-allocations['graham']-allocations['stingy']-allocations['cash']
        allocations_g += [allocations]
        # simulate 
        r_dict = simulate_portfolio(df_returns,
                           allocations=allocations,
                           initial_investment=initial_investment, 
                           annual_adds=annual_adds,
                           risk_free_rate=risk_free_rate, 
                           borrowed_interest=borrowed_interest,
                           min_cash=min_cash)
        #
        results += [r_dict]
    
    return results
    

'''
helper function to transform data for contour plot
'''
def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = meshgrid(xi, yi)
    return X, Y, Z



def plot_ratios(many_simulation_results,
                x_data='graham',
                y_data='stingy',
                z_data='sharpe_ratio',
                x_tick_spacing=0.1,
                y_tick_spacing=0.1):
    
    # create data arrays for plotting
    x_values = [x['target_allocation'][x_data] for x in many_simulation_results]
    y_values = [x['target_allocation'][y_data] for x in many_simulation_results]
    z_values = [x[z_data] for x in many_simulation_results]
    X, Y, Z = grid(x_values, y_values, z_values)
    
    # setup fiture
    fig = plt.figure(figsize=(10,7.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(x_data)
    ax.set_ylabel(y_data)
    ax.grid()
    
    # plot values
    cf = ax.contourf(X, Y, Z)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(z_data)
    
    lower=2/100
    upper=99/100
    # adjust x_label properties
    x_min = lower*(max(x_values)-min(x_values)+min(x_values))
    x_max = upper*(max(x_values)-min(x_values)+min(x_values))
    plt.xlim(x_min, x_max)
    plt.xticks([x_min]+[x*x_tick_spacing for x in range(int(np.ceil(x_min/x_tick_spacing)),
                                                int(np.floor(x_max/x_tick_spacing))+1)]+[x_max])
    ax.set_xticklabels(['{:.2f}'.format(x) for x in ax.get_xticks().tolist()])
        
        
    y_min = lower*(max(y_values)-min(y_values)+min(y_values))
    y_max = upper*(max(y_values)-min(y_values)+min(y_values))
    plt.ylim(y_min, y_max)
    plt.yticks([y_min]+[x*y_tick_spacing for x in range(int(np.ceil(y_min/y_tick_spacing)),
                                                int(np.floor(y_max/y_tick_spacing))+1)]+[y_max])
    ax.set_yticklabels(['{:.2f}'.format(x) for x in ax.get_yticks().tolist()])
    
    plt.show()
    
    return
    



def main():
    pass

if __name__ == "__main__":
    main()