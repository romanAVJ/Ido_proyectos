# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:46:45 2020

@author: Roman AVJ
"""
import numpy as np
import pandas as pd

import six # needed to import mlrose
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as mr

def times_array(file_time, file_prod):
    #read production demand time
    prod_time = np.genfromtxt(file_prod, delimiter=',')
    
    # read times from passing one color to other and parse to numpy array
    df = pd.read_csv(file_time, header=0, dtype=float)
    
    # get lipstick names
    lipstick = list(df.columns.values)
    
    # as numpy arrray
    X = df.to_numpy(dtype=float)
    X[X == 0] = np.finfo(float).eps # very little number
    n = len(X)
    
    # get upper triangular valueswithout diagonal as an array
    # iu = np.triu_indices(n, 1)
    # Xup = X[iu]
    
    # list of distances between cities
    dist_list = []
    for i in range(n-1):
        for j in range(i+1, n):
            dist_list.append((i,j, X[i,j]))
            
                
      
    return(dist_list, lipstick, prod_time)

def metaheuristic(dist, meta=None, nnodes=None, **kwargs):
    # fit data
    fit_dist = mr.TravellingSales(distances=dist)
    
    # define TSP 
    fit_problem = mr.TSPOpt(length=nnodes, fitness_fn=fit_dist, maximize=False)
    
    # define optmization metaheuristic
    # kwargs are parameters to the metaheuristic
    best_state, best_fit = meta(fit_problem, **kwargs)
    
    return(best_state, best_fit)

def meta_ensamble(dist, n, kwargs_ga={}, kwargs_sa={}, iters=10):
    """
    Metaheuristic ensambles. It ensambles a Genetic Algorithm and then uses
    the best state of the latter as a initial state in a Simulated Anealing 
    Algorithm. 
    The idea is taken form  Hillier, Lieberman.
    Both metaheuristics are implemented by the 'mlrose' package
    Parameters
    ----------
    dist : list
        list of tribles. The first to elements are the (i,j) nodes and the
        last element is the distance d(i,j)
    n : int
        Number of cities.
    kwargs_ga : dic, optional
        Optional arguments to the mlrose Genetic Alg. The default is {}.
    kwargs_sa : TYPE, optional
        Optional arguments to the mlrose Sim Anealing. The default is {}.
    iters : int, optional
        Number of iterations of the ensamble. The default is 10.

    Returns
    -------
    champion_fit : float
        The best fit found.
    champion_state : narray
        Numpy array with the orders of the nodes of the TSP.

    """
    #init 
    best_states = []
    best_fitness = {}
    
    for i in range(iters):
        # fix randomness
        kwargs_ga['random_state'] = i
        kwargs_sa['random_state'] = i
        
        # use genetic algorithm
        state_ga, _ = metaheuristic(dist, meta=mr.genetic_alg, nnodes=n, **kwargs_ga)
        
        # use simulated anealing with the solution of the previous GA
        kwargs_sa['init_state'] = state_ga
        best_state, best_fit = metaheuristic(times_list, meta=mr.simulated_annealing, 
                                                 nnodes=n, **kwargs_sa)
        
        #append sols
        best_states.append(best_state)
        best_fitness[best_fit] = i
    
    # get best solution of all
    champion_fit = min(best_fitness)
    best_iter = best_fitness[champion_fit]
    champion_state = best_states[best_iter]
        
    return(champion_state, champion_fit)
    
def pruebas():
    # read data
    print('\n============== Problema del agente viajero ======================')
    #### get data
    times_list, lips, prod_lips = times_array('datos_macbelle_dummie.csv', 'tiempos_produccion.csv')
    
    #### metaheuristics
    n = len(lips)
    
    #genetic alg
    # simple
    args = {'random_state': 41}
    bs, bf = metaheuristic(times_list, meta=mr.genetic_alg, nnodes=n, **args)
    
    # modified hypérparameters
    args2 = {'random_state': 41, 'mutation_prob': 0.3, 'max_attempts': 20}
    bs2, bf2 = metaheuristic(times_list, meta=mr.genetic_alg, nnodes=n, **args2)
    
    # simulated anealing
    # simple
    args_sa = {'random_state': 41}
    bs_sa, bf_sa = metaheuristic(times_list, meta=mr.simulated_annealing, nnodes=n, **args_sa)
    
    # change temperature decay
    # look documentation mlrose of decays 
    args_sa2 = {'random_state': 41, 'schedule': mr.ExpDecay(exp_const=0.05)}
    bs_sa2, bf_sa2 = metaheuristic(times_list, meta=mr.simulated_annealing, nnodes=n, **args_sa2)
    
    # combined with genetic alg solutions
    args_sa3 = {'random_state': 41, 'schedule': mr.GeomDecay(decay=0.05), 
                'init_state': bs} #use GA solution as initial state
    bs_sa3, bf_sa3 = metaheuristic(times_list, meta=mr.simulated_annealing, nnodes=n, **args_sa3)
    return()
    

# =============================================================================
# MAIN
# =============================================================================
# read data
print('\n============== Problema del agente viajero ======================')
#### get data
times_list, lips, prod_lips = times_array('datos_macbelle_dummie.csv', 'tiempos_produccion.csv')

#### metaheuristics
n = len(lips)
args_ga = {'mutation_prob': 0.1, 'max_attempts': 20}
args_sa = {'schedule': mr.ExpDecay(exp_const=0.05)}

bests, bestf = meta_ensamble(times_list, n, args_ga, args_sa, iters=2)













