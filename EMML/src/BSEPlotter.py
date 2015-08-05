'''
Created on 8 Apr 2014

@author: asbh500
'''

import time, datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import BSE
from shutil import copyfile
from scipy.stats import wilcoxon, mannwhitneyu
from timeit import itertools



## returns a tuple with the output files

def _launchExperiment(timestamp, traders_type, n_traders_per_type, n_trials_per_ratio, marketSessionTime,
                      supplyTFunc, demandTFunc, dumpEachTrade):
    # set up parameters for the session
    # TODO consider to move to experiment parameter
    start_time = 0.0
    supplyStartValue = 90
    supplyStepMode = 'fixed'
    demandStartValue = 110
    demandStepModel = 'fixed'
    replenishmentInterval = 30
    shedulerTimeMode = 'drip-poisson'

    def defaultFunc(t):
        pi2 = math.pi * 2
        c = math.pi*3000
        wavelength = t/c
        gradient = 100*t/(c/pi2)
        amplitude = 100*t/(c/pi2)
        offset = gradient + amplitude * math.sin(wavelength*t)
        return int(round(offset,0))
    # tFunc = time-dependent offset on schedule prices
    if (supplyTFunc == None): supplyTFunc = defaultFunc
    if (demandTFunc == None): demandTFunc = defaultFunc


    range1 = (supplyStartValue, supplyStartValue, supplyTFunc)
    supply_schedule = [ {'from':start_time, 'to':marketSessionTime, 'ranges':[range1], 'stepmode':supplyStepMode}
                      ]

    range1 = (demandStartValue, demandStartValue, demandTFunc)
    demand_schedule = [ {'from':start_time, 'to':marketSessionTime, 'ranges':[range1], 'stepmode':demandStepModel}
                      ]

    order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
                   'interval':replenishmentInterval, 'timemode': shedulerTimeMode}

    n_trader_types = len(traders_type)
    n_traders = n_trader_types * n_traders_per_type

    # Output files
    configFileName = '%s_config.txt' % (timestamp)
    configFile = open(configFileName, 'w')
    configFile.writelines([
        'Trader types = {}\n'.format(traders_type),
        'Number of traders per type= {}\n'.format(n_traders_per_type),
        'Number of trials per traders ratio= {}\n'.format(n_trials_per_ratio),
        'Supply start value = {}\n'.format(supplyStartValue),
        'Supply step mode = {}\n'.format(supplyStepMode),
        'Supply function = {}\n'.format(supplyTFunc.__name__),
        'Demand start value = {}\n'.format(demandStartValue),
        'Demand step mode = {}\n'.format(demandStepModel),
        'Demand function = {}\n'.format(demandTFunc.__name__),
        'Replenishment interval = {}\n'.format(replenishmentInterval),
        'Scheduler time mode= {}\n'.format(shedulerTimeMode),
        'Market session open time={}\n'.format(start_time),
        'Market session close time= {}\n'.format(marketSessionTime),
        'Dump each trace = {}\n'.format(dumpEachTrade),
        ])
    configFile.close()

    tdumpName = '%s_balances.csv' % (timestamp)
    tdump=open(tdumpName,'w')


    simParameters = {}
    simParameters['start_time'] = start_time
    simParameters['marketSessionTime'] = marketSessionTime
    simParameters['n_trials_per_ratio'] = n_trials_per_ratio
    simParameters['order_sched'] = order_sched
    simParameters['tdump'] = tdump
    simParameters['dumpEachTrade'] = dumpEachTrade

    _runMarketSession(1, n_traders, traders_type, 0, {}, 1, simParameters);

    # we close the balances file and copy the transactions one
    tdump.close()
    copyfile('transactions.csv', '{}_transactions.csv'.format(timestamp)) # Useful, just for one trial experiments
    return [tdumpName , 'transactions.csv']

def _runMarketSession(trialnumber, n_traders, trdr_types, trdr_type_it, previous_trdr_counters, min_n, simParameters ):

    trdr_type = trdr_types[trdr_type_it];
    current_trdr_counters = previous_trdr_counters.copy()
    if trdr_type_it == len(trdr_types)-1:
        current_trdr_counters[trdr_type] = n_traders - sum(previous_trdr_counters.values())
        if current_trdr_counters[trdr_type] >= min_n:
            buyers_spec = []
            for trdr_type in trdr_types:
                buyers_spec.append((trdr_type, current_trdr_counters[trdr_type]))

            sellers_spec = buyers_spec
            traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
            #print buyers_spec
            trial = 1
            tdump = simParameters['tdump']
            while trial <= simParameters['n_trials_per_ratio']:
                    trial_id = 'trial%07d' % trialnumber
                    BSE.market_session(trial_id, simParameters['start_time'], simParameters['marketSessionTime'], traders_spec,
                                   simParameters['order_sched'], tdump, simParameters['dumpEachTrade'] )
                    tdump.flush()
                    trial = trial + 1
                    trialnumber = trialnumber + 1
    else :
        current_trdr_counters[trdr_type] =  min_n
        while (current_trdr_counters[trdr_type] <= (n_traders - sum(previous_trdr_counters.values()))):
            _runMarketSession(trialnumber,n_traders, trdr_types, trdr_type_it + 1, current_trdr_counters, min_n, simParameters)
            current_trdr_counters[trdr_type] = current_trdr_counters[trdr_type] + 1; # increment the number of traders

def _saveResults(timestamp, balancesFName, n_trader_types, n_trials_per_ratio, iterativelyShow=False):

    traders_offset = 2
    traders_fields = 4
    traders_numberOfRobots = 2
    trader_balancePerTrader_offset = 3


    traderTypesCols = []
    averageProfitCols = []
    numberOfTradersCols = []
    for i in xrange(n_trader_types):
        averageProfitCols.append(traders_offset + traders_fields*i + trader_balancePerTrader_offset)
        traderTypesCols.append(traders_offset + traders_fields*i)
        numberOfTradersCols.append(traders_offset + traders_fields*i + traders_numberOfRobots)

    allAvgProfits = np.loadtxt(balancesFName, delimiter=',', usecols=averageProfitCols, unpack = True)
    allTraderTypes =  np.loadtxt(balancesFName, delimiter=',', usecols=traderTypesCols, unpack = True, dtype = np.str)
    allNumOfTraders=  np.loadtxt(balancesFName, delimiter=',', usecols=numberOfTradersCols, unpack = True, dtype = np.str)
    if n_trader_types == 1: # If we have only one type of traders np.loadtxt will give us a simple
                            # array instead of an array of arrays
        allAvgProfits = [allAvgProfits]
        allTraderTypes = [allTraderTypes]
        allNumOfTraders = [allNumOfTraders]

    # compute the list of traders types
    traders_types = []
    for i in xrange(len(allTraderTypes)):
        traders_types.append(allTraderTypes[i][0]) # we only need the first one


    resultsFileName = '{}_results.txt'.format(timestamp)
    resultsFile=open(resultsFileName,'w')

    # compute means, of each ratio and total
    groupAvgProfitMeans = []
    totalAvgProfitMeans = []
    for i in xrange(len(allAvgProfits)): # For each trader type
        balancePerTraders = allAvgProfits[i]
        nTraders = allNumOfTraders[i]
        totalAvgProfitMeans.append(np.mean(balancePerTraders))
        resultsFile.write("{0}:\nTotal mean: {1}\n".format(traders_types[i], totalAvgProfitMeans[i]))
        groupAvgProfitMeans.append([])
        for j in xrange(0, len(balancePerTraders), n_trials_per_ratio): # For each experiment group
            groupAvgProfits = balancePerTraders[j:j+n_trials_per_ratio]
            numberOfRobots= nTraders[j:j+n_trials_per_ratio][0]
            mean = np.mean(groupAvgProfits)
            sd = np.std(groupAvgProfits)
            resultsFile.write("{0} robots: mean = {1}, stdev = {2} \n".format(numberOfRobots, mean, sd))
            groupAvgProfitMeans[i].append(mean)


    # We configure the plot
    plt.clf() # We clear the plot
    plt.cla()
    ax = plt.subplot(111)

    xMin = 1
    xMax = len(groupAvgProfitMeans[0]) + 1

    xAxis = xrange(xMin,xMax)
    xTicks = []
    step = round(len(xAxis) / 7) # 7 ticks
    for i in xAxis:
        if (i % step == 0):
            xTicks.append('G{}'.format(i))
        else:
            xTicks.append('')
    plt.xticks(xAxis, xTicks)


    for i in xrange(len(groupAvgProfitMeans)):
        array = groupAvgProfitMeans[i]
        plot, = plt.plot(xAxis, array, label=traders_types[i])
        plt.plot(xAxis, np.repeat(totalAvgProfitMeans[i], len(array)), color=plot.get_color(), label='Mean: {0:.2f}'.format(totalAvgProfitMeans[i]), linestyle='dashed')
    plt.title('Average profit per experiment group')
    plt.xlabel("Experiment Groups")
    plt.ylabel("Average Profit")

    # resize and set the legnd
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig("{}_plot_a.pdf".format(timestamp))

    #resultsFile.write("All data Mann-Whitney U-test:\n");
    #for subset in itertools.combinations(xrange(0, len(allAvgProfits)), 2):
    #    i, j = subset
    #    u, p_value = mannwhitneyu(allAvgProfits[i], allAvgProfits[j])
    #    resultsFile.write("{0} - {1} : u = {2}, p-value={3}\n".format(traders_types[i], traders_types[j], u, p_value))

    resultsFile.write("Experiment group means Mann-Whitney U-test:\n");
    for subset in itertools.combinations(xrange(0, len(groupAvgProfitMeans)), 2):
        i, j = subset
        u, p_value = mannwhitneyu(groupAvgProfitMeans[i], groupAvgProfitMeans[j])
        resultsFile.write("{0} - {1} : u = {2}, p-value={3}\n".format(traders_types[i], traders_types[j], u, p_value))

    #resultsFile.write("Wilcoxon test:\n");
    #for subset in itertools.combinations(xrange(0, len(allAvgProfits)), 2):
    #    i, j = subset
    #    t, p_value = wilcoxon(allAvgProfits[i], list(reversed(allAvgProfits[j])))
    #    resultsFile.write("{0} - {1} : t = {2}, p-value={3}\n".format(traders_types[i], traders_types[j], t, p_value))

    resultsFile.close()
    if iterativelyShow :
        plt.show()

def __plotSupplyDemandFunction__():

    def tFunc(t):
        pi2 = math.pi * 2
        c = math.pi*3000
        wavelength = t/c
        gradient = 100*t/(c/pi2)
        amplitude = 100*t/(c/pi2)
        offset = (gradient + amplitude * np.sin(wavelength*t))
        return int(round(offset,0))

    def tFunc2(t):
        pi2 = math.pi * 2
        c = math.pi*3000
        wavelength = t/c
        gradient = 100*t/(c/pi2)
        amplitude = 100*t/(c/pi2)
        offset = - (gradient + amplitude * np.sin(wavelength*t))
        return int(round(offset,0))

    x = np.arange(450) # 100 linearly spaced numbers

    supplies = []
    demands =[]
    for i in xrange(len(x)):
        supplies.append(90 + tFunc(x[i]))
        demands.append(110 + tFunc(x[i]))
    plt.plot(x, supplies, label="Supply")
    #plt.plot(x, demands, label="Demand")
    #plt.legend(loc="upper center")
    plt.show()

def main():

    #__plotSupplyDemandFunction__()

    market_session = 450.0
    launchExperiment(['SZIP', 'ZIP'], marketSessionTime=market_session)
    launchExperiment(['SZIP', 'ZIP', 'SHVR'], marketSessionTime=market_session)


def launchExperiment(traders_type=['GVWY','SHVR','ZIC','ZIP'], n_traders_per_type = 4, n_trials_per_ratio = 50, marketSessionTime=600.0, dumpEachTrade = False,
                     supplyFunc = None, demandFunc = None):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
    resultFiles = _launchExperiment(timestamp, traders_type, n_traders_per_type, n_trials_per_ratio, marketSessionTime, supplyFunc, demandFunc, dumpEachTrade)
    _saveResults(timestamp,  resultFiles[0], len(traders_type), n_trials_per_ratio)

def launchExperimentAndShowPlot(traders_type=['GVWY','SHVR','ZIC','ZIP'], n_traders_per_type = 4, n_trials_per_ratio = 50, marketSessionTime=600.0, dumpEachTrade = False,
                                supplyFunc = None, demandFunc = None ):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H%M%S')
    resultFiles = _launchExperiment(timestamp, traders_type, n_traders_per_type, n_trials_per_ratio, marketSessionTime, supplyFunc, demandFunc, dumpEachTrade)
    _saveResults(timestamp, resultFiles[0], len(traders_type), n_trials_per_ratio)

if __name__ == '__main__':
    main()