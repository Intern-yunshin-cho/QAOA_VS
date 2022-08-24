import kqc_custom
import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit import Aer,IBMQ
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    RecursiveMinimumEigenOptimizer,
    SolutionSample,
    OptimizationResultStatus,
)
from qiskit.visualization import plot_histogram
from typing import List, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import winsound
import time

'''
    repo 의 노트북 실행을 가정 
    나중에 코드 정리되면 한번에 vectorization 수정하자
'''
def projection(X):
    '''
        projection
    '''
    xtx = np.matmul(X.T,X)
    xtxinv = np.linalg.inv(xtx)
    proj = np.matmul(np.matmul(X,xtxinv),X.T)
    return(proj)
    
def partial_r1(X,y):
    '''
        ind : 
        SSRR : 
    '''
    X = np.asarray(X);y=np.asarray(y).reshape((-1))
    n = X.shape[0]
    X_temp = pd.DataFrame(np.ones((n,1))); X = pd.DataFrame(X)
    X_temp = np.asarray(pd.concat([X_temp,X],axis=1))
    q=X_temp.shape[1]
    y = y-np.mean(y)
    partial_r_list = []
    SSRF = np.matmul(np.matmul(y.T,(np.identity(n)-projection(X_temp))),y)
    for i in range(1,q):
        ind = [i for i in range(i)] + [i for i in range(i+1,q)]
        SSRR = np.matmul(np.matmul(y.T,(np.identity(n)-projection(X_temp[:,ind]))),y)
        partial_r_list += [1-SSRF/SSRR]
    return(np.array(partial_r_list))

def partial_r2(X,y):
    '''
        ind : 
        SSRR : 
    '''   
    n = X.shape[0]
    X_temp = pd.DataFrame(np.ones((n,1))); X = pd.DataFrame(X)
    X_temp = np.asarray(pd.concat([X_temp,X],axis=1)) ; y = np.asarray(y)
    p=X_temp.shape[1]
    y = y-np.mean(y)
    partial_r_list = []
    SSRF = np.matmul(np.matmul(y.T,(np.identity(n)-projection(X_temp))),y)
    for i in range(1,p):
        SSRR = np.matmul(np.matmul(y.T,(np.identity(n)-projection(X_temp[:,[0,i]]))),y)
        partial_r_list += [SSRF/SSRR]
    return(np.array(partial_r_list))

def qaoa(x,y,backend = Aer.get_backend("qasm_simulator")) :
    # gamma = default로 2 ? 1로?
    data_x = pd.DataFrame(x)
    data_y = pd.DataFrame(y)
    p = data_x.shape[1]
    Q = np.abs(data_x.corr())
    for i in range(p) : Q.iloc[i,i] = 0 
    r_squared_list=partial_r1(data_x,data_y)
    
    beta_new =  -Q.apply(sum,axis=1) -2*r_squared_list
    result_qaoa=kqc_custom.qubo_qaoa(Q,beta_new,backend)
    return(result_qaoa)