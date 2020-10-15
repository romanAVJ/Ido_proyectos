#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:16:53 2020

@author: RomanAVJ
"""

import pandas as pd
import numpy as np

import pulp as pl

from itertools import product # cartesian product
from ast import literal_eval as make_tuple
from time import time


import six # needed to import mlrose
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose as mr


def read_cities(file):
    return pd.read_csv(file, index_col = 0, header = None,dtype = float).to_numpy()
    
def distance(file):
    # read data
    coords = read_cities(file)
    
    # values
    n = len(coords)
    dist_list = []
    
    # write tuples of nodes and distance
    for i in range(n-1):
        for j in range(i+1,n):
            dij = np.linalg.norm(coords[i]-coords[j])
            dist_list.append((i, j, dij))
            
    # maybe isnt symetrical
    # for i in range(n):
    #     for j in range(n):
    #         if i != j:
    #             dij = np.linalg.norm(coords[i]-coords[j])
    #             dist_list.append((i, j, dij))

    return (dist_list, n)

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
        kwargs_ga['random_state'] = 41 + i
        kwargs_sa['random_state'] = 41 + i
        
        # change mutation (more)
        kwargs_ga['mutation_prob'] = 0.1 + 0.05*i
        # use genetic algorithm
        state_ga, _ = metaheuristic(dist, meta=mr.genetic_alg, nnodes=n, **kwargs_ga)
        
        # use simulated anealing with the solution of the previous GA
        kwargs_sa['init_state'] = state_ga
        best_state, best_fit = metaheuristic(dist, meta=mr.simulated_annealing, 
                                                 nnodes=n, **kwargs_sa)
        
        #append sols
        best_states.append(best_state)
        best_fitness[best_fit] = i
    
    # get best solution of all
    champion_fit = min(best_fitness)
    best_iter = best_fitness[champion_fit]
    champion_state = best_states[best_iter]
        
    return(champion_state, champion_fit)

def eval_meta(file, meta, args_meta={}):
    # get nodes and distance
    dist, nnodes = distance(file)
    
    # evaluate and get best fit
    begin = time()
    bs, bf = metaheuristic(dist, meta=meta, nnodes=nnodes, **args_meta)
    time_meta = time() - begin
    
    return(file, bf, time_meta)
    
def df_meta(meta, files, m_args = {}, meta_names = []):
    n = len(m_args)
    df = pd.DataFrame(columns=['model', 'file','fit', 'time'])
    
    # look if the same number of names are the same number of args
    if n != len(meta_names):
        raise Exception('ERROR: meta names must be same length that m args')
        
    # loop 
    j = 0
    for i in range(n):
        for file in files:
            df.loc[j, ['file','fit', 'time']] = eval_meta(file, meta, args_meta=m_args[i])
            df.loc[j, 'model'] = meta_names[i]
            print('End process for ' + file)
            j += 1
        print('\n\t End all files for one meta\n')
    
    return(df)
    
def df_ensamble(files, argsGA={}, argsSA={}):

    #df 
    df = pd.DataFrame(columns=['file','fit', 'time'])
    
    
    # evaluate and get best fit
    i = 0
    for file in files:
        # get nodes and distance
        dist, nnodes = distance(file)
    
        # take time
        begin = time()
        
        # eval
        bs, bf = meta_ensamble(dist, nnodes, kwargs_ga=argsGA, kwargs_sa=argsSA, iters=5)
        time_meta = time() - begin
        
        # save data
        df.loc[i] = (file, bf, time_meta)
        
        print('End metaensamble process for ' + file)
        i += 1
        
    return(df)


# =============================================================================
#               MAIN
# =============================================================================

# initial values
files = ["att48.csv","bayg29.csv","berlin52.csv","eil51.csv","st70.csv"]
files = ["bayg29.csv", "berlin52.csv"]
file = "bayg29.csv"
seed = 41
namesSA = ['modelo_caliente', 'modelo_frio']
namesGA = ['modelo_XXgrande', 'modelo_XYchico']
argsSA = [
        # geometric decay
            {'random_state': seed,
             'max_attempts': 10,#ojo!!!
             'schedule': mr.GeomDecay(decay = 0.9)},
            {'random_state': seed,
              'max_attempts': 1000,
              'schedule': mr.GeomDecay(decay = 0.99)},
        #     {'random_state': seed,
        #      'max_attempts': 30,
        #      'schedule': mr.GeomDecay()},
        
        # # arithmetic
        #     {'random_state': seed,
        #      'max_attempts': 50,
        #      'schedule': mr.ArithDecay(decay = 0.1)},
        #     {'random_state': seed,
        #      'max_attempts': 15,
        #      'schedule': mr.ArithDecay(decay = 0.05)},
        #     {'random_state': seed,
        #      'max_attempts': 15,
        #      'schedule': mr.ArithDecay()},
            
        # # exponencial
        #     {'random_state': seed,
        #       'max_attempts': 15,
        #       'schedule': mr.ExpDecay(exp_const = 0.01)},
        #     {'random_state': seed,
        #       'max_attempts': 15,
        #       'schedule': mr.ExpDecay(exp_const = 0.05)},
        #     {'random_state': seed,
        #       'max_attempts': 30,
        #       'schedule': mr.ExpDecay()},
        ]
argsGA = [
        # same size
            {'random_state': seed,
             'pop_size': 200,
             'mutation_prob': 0.4, # mejora 
             'max_attempts': 1000},
        #     {'random_state': seed,
        #      'pop_size': 100,
        #      'max_attempts': 15},
        #     {'random_state': seed,
        #      'pop_size': 100,
        #      'mutation_prob': 0.2,
        #      'max_attempts': 15},
            
        # # mutatetion
        #     {'random_state': seed,
        #      'pop_size': 100,
        #      'mutation_prob': 0.15,
        #      'max_attempts': 15},
        #     {'random_state': seed,
        #      'pop_size': 200,
        #       'mutation_prob': 0.15,
        #      'max_attempts': 15},
        #     {'random_state': seed,
        #      'pop_size': 300,
        #      'mutation_prob': 0.15,
        #      'max_attempts': 15},
        ]

# args
# marginal
argsGA =  [{'random_state': seed,
             'pop_size': 30,
             'mutation_prob': 0.1, 
             'max_attempts': 200},
             {'random_state': seed,
             'pop_size': 20,
             'mutation_prob': 0.3,
             'max_attempts': 200}
             ]

argsSA=  [{'random_state': seed,
              'max_attempts': 300,
              'schedule': mr.GeomDecay(decay = 0.9)},
             {'random_state': seed,
              'max_attempts': 500,
              'schedule': mr.GeomDecay(decay = 0.7, min_temp=1e-8)
             }
          ]

#ensamble
argsGA_a =  {'pop_size': 20,
             'mutation_prob': 0.15, 
             'max_attempts': 100
             }

argsSA_a =  {'max_attempts': 150,
              'schedule': mr.GeomDecay(decay = 0.9)}


#### ops
print("==========================================================\n" +
       "\t marginal .metheuristics")
df_sa = df_meta(mr.simulated_annealing, files, m_args=argsSA, meta_names=namesSA)
# df_ga = df_meta(mr.genetic_alg, files, m_args=argsGA, meta_names=namesGA)

print("\n==========================================================\n" +
      "\t ensamble")
      
      
# asamble
# df_ens = df_ensamble(files, argsGA=argsGA_a, argsSA=argsSA_a)
