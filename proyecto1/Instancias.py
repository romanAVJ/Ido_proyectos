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

def eval_meta(file, meta, args_meta={}):
    # get nodes and distance
    dist, nnodes = distance(file)
    
    # evaluate and get best fit
    begin = time()
    bs, bf = metaheuristic(dist, meta=meta, nnodes=nnodes, **args_meta)
    time_meta = time() - begin
    
    return(bf, time_meta)
    
def df_meta(meta, file, m_args = {}, meta_names = []):
    n = len(m_args)
    df = pd.DataFrame(index=meta_names, columns=['fit', 'time'])
    
    # look if the same number of names are the same number of args
    if n != len(meta_names):
        raise Exception('ERROR: meta names must be same length that m args')
        
    # loop 
    for i in range(n):
        df.loc[meta_names[i]] = eval_meta(file, meta, args_meta=m_args[i])
    
    return(df)
    



# =============================================================================
#               MAIN
# =============================================================================

# initial values
# nombres = ["att48.csv","bayg29.csv","berlin52.csv","eil51.csv","st70.csv"]
file = "bayg29.csv"
seed = 41
namesSA = ['modelo' + str(x) for x in range(1,2)]
namesGA = ['modelo' + str(x) for x in range(1,2)]
args = [
        # geometric decay
            {'random_state': seed,
             'max_attempts': 100,#ojo!!!
             'schedule': mr.GeomDecay(decay = 0.9)},
        #     {'random_state': seed,
        #      'max_attempts': 100,
        #      'schedule': mr.GeomDecay(decay = 0.6)},
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
             'pop_size': 100,
             'mutation_prob': 0.4,
             'max_attempts': 100},
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


# look model annealing
df_sa = df_meta(mr.simulated_annealing, file, m_args=args, meta_names=namesSA)
df_ga = df_meta(mr.genetic_alg, file, m_args=argsGA, meta_names=namesGA)





































