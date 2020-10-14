# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:34:34 2020

@author: RomanAVJ
"""
import numpy as np
import pandas as pd

import pulp as pl
from itertools import product # cartesian product
from ast import literal_eval as make_tuple

def times_matrix(file_time, file_prod):
    #read production demand time
    prod_time = np.genfromtxt(file_prod, delimiter=',')
    
    # read times from passing one color to other and parse to numpy array
    df = pd.read_csv(file_time, header=0, dtype=float)
    
    # get lipstick names
    lipstick = list(df.columns.values)
    
    # as numpy arrray
    X = df.to_numpy()
    return(X, lipstick, prod_time)

def TSP(T, k, names):
    # initialize ILP as a min problem
    model = pl.LpProblem(name='Optimizacion_Labiales', sense=pl.LpMinimize)
    
    # number of lipsticks
    n = len(T)
    N = {x for x in range(1, n + 1)}
    
    print('\nNúmero de productos a optimizar sus tiempos de producción: ', (n-1))
        
    # variables
    x = pl.LpVariable.dicts('cambio_labial',
                            ((i, j) for i in N for j in N),
                            cat='Binary'
                            )
    
    # objective function
    z = pl.LpAffineExpression(e=[(x[i,j], T[i-1,j-1]) for i,j in x], 
                              constant=k,
                              name='Funcion_Objetivo'
                              )    
    model += z
    
    # constraints
    # constraint 1: all are visited but only once
    for i in N:
        N_update = N - {i}
        model += pl.LpConstraint([(x[i,j], 1) for j in N_update],
                                 sense=pl.LpConstraintEQ,
                                 rhs=1 # right side of the equality
                                 )
    
    # constraint 2: all are leaved and only once
    for j in N:
        N_update = N - {j}
        model += pl.LpConstraint([(x[i,j], 1) for i in N_update],
                         sense=pl.LpConstraintEQ,
                         rhs=1 # right side of the equality
                         )
        
    # subtour elimination
    # aux vars definied by Miller, Tucker and Zemlin
    u = pl.LpVariable.dict('aux_var',
                           (i for i in N),
                           cat='Continuous'
                           )
    
    N0 = N - {1}
    for (i, j) in product(N0, N0):
        if i != j:
            model += pl.LpConstraint(u[i] - u[j] + n*x[i,j],
                                     sense=pl.LpConstraintLE,
                                     rhs = n-1
                                     )    
    # solve model
    # solve
    model.solve() 

    # print answer
    print_TSP(model, names, k, n)
    
    return()

def print_TSP(model, names, k, n):
    
    if model.status == 1:
        # print objective function
        print('\nEl tiempo de producción óptimo es de:', model.objective.value())
        print('\t\t\t Tiempo total demandado fijo:', k)
        print('\t\t\t Tiempo total de pasar de un producto a otro:', (model.objective.value() - k))
        
        #print decision vars
        print('\n\t*********** Secuencia de productos ***********')
        nodes = get_vars(model, n)
        for node in nodes:
            print(names[node - 1], '-->')
        
    else:
        print('\nNo se alcanzó el óptimo, checar datos y formulación')
        
def get_vars(model, n):
    # get decision vars only (i,j)
    t_raw = [x.name.replace('_','')[12:] for x in model.variables() if x.name[0:3] != 'aux' and x.value() == 1]
    
    #make tuple
    x = [make_tuple(t) for t in t_raw]

    # create printable sequence
    # look dummie var, i.e. last n var
    i = 0
    node_list = []
    
    while x[i][0] != n:
        i += 1
    # node where to go and save it
    yi = x[i][1] 
    node_list.append(yi)

    # do it for all the list n-1 except the dummie node
    for _ in range(n-2):
        j = 0
        while x[j][0] != yi:
            j += 1
        yi = x[j][1]
        node_list.append(yi)

    return(node_list)

# =============================================================================
# MAIN
# =============================================================================
# print('\n============== Programación óptima de producción ======================')
# #### get data
# T_times, lips, prod_lips = times_matrix('datos_macbelle_dummie.csv', 'tiempos_produccion.csv')

# #### formulation of TSP
# # constant
# k = sum(prod_lips)

# # tsp
# TSP(T_times, k, lips)

# print('\nbye 👋')

