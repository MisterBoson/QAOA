#General Packages
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import ipyparallel as ipp
plt.style.use('classic')

import pennylane as qml

#Qiskit packages
import qiskit
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, ADAM, SLSQP, GradientDescent
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import I, X, Y, Z, Zero, One, PauliExpectation, CircuitSampler, StateFn, DictStateFn, CircuitStateFn, NaturalGradient
import qaoa_functions as fn




main_path= r'/home/mrboson/Documents/Academic_Route/MS_Physics/Courses/Computational Physics /qaoa_repo/quantum-approximate-optimization-algorithm'

from multiprocessing import Pool
import os

if __name__ == '__main__':
    path_wd = path_parent # r'C:\Users\manue\Desktop\UNI\9. Semester\CP-2'
    while not (os.getcwd()==path_wd):
        os.chdir(path_wd)
    # from multiprocessing_function import run_qaoa_dict, RandomIsing_Manuel
    
    
    
    
    path_network = path_wd + r'/Networks/'
    with open(path_network + '0 G, degree=4, order=5, localTerm=True' +'.txt', 'rb') as my_file:
        G = nx.read_adjlist(my_file)
    
    J = np.load(path_network + '0 J, degree=4, order=5, localTerm=True.npy', allow_pickle=True).item()
    h = np.load(path_network + '0 h, degree=4, order=5, localTerm=True.npy')
    
    nx.draw(G)
    
    def check_params(param, n_repititions):
        if np.isscalar(param):
            return np.repeat(param, n_repititions)
        elif len(param)==n_repititions:
            return np.array(param)
        else:
            raise ValueError('Shape of param does not match requirements')
    
    # Parameters
    parent_path = path_wd + r'/Results/'
    n_repititions = 1
    n_parallel = 8
    max_iterations = 3 # Standard value, can be changed when called from console
    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print(sys.argv[1])
        if not sys.argv[1] == '':
            max_iterations = int(sys.argv[1])
    print('max_iterations: ', max_iterations)
    
    #Constants
    method_rep    = ['QNGD', 'AdamGD']
    degree_rep    = 4  # Number of connected nodes
    order_rep     = [5,6,8,10]  # Number of Qubits
    depth_rep     = 5
    step_size_rep = 0.01
    localTerm_rep = True
    random_graph_rep = False
    
    method_rep = check_params(method_rep, n_repititions)
    degree_rep = check_params(degree_rep, n_repititions)
    order_rep = check_params(order_rep, n_repititions)
    depth_rep = check_params(depth_rep, n_repititions)
    step_size_rep = check_params(step_size_rep, n_repititions)
    localTerm_rep = check_params(localTerm_rep, n_repititions)
    random_graph_rep = check_params(random_graph_rep, n_repititions)
    
    
    for i in range(n_repititions):
        method = method_rep[i].item()
        degree = degree_rep[i].item()
        order = order_rep[i].item()
        depth = depth_rep[i].item()
        step_size = step_size_rep[i].item()
        localTerm = localTerm_rep[i].item()
        random_graph = random_graph_rep[i].item()
        print('method: ', method)
        print('depth: ', depth)
        folder_addition = '_test'
        path = parent_path + datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + folder_addition
        os.makedirs(path)
        args = []
        for i in range(n_parallel):
            args_i = {'process_number':  i, 
                      'path':            path,
                      'method':          method, 
                      'd':               degree, 
                      'nqubits':         order, 
                      'p':               depth, 
                      'max_iterations':  max_iterations,
                      'localTerm':       localTerm,
                      'step_size':       step_size,
                      }
            if not random_graph:
                args_i.update({'G': G.copy(), 'J': J.copy(), 'h': h.copy()})
            args.append(args_i)
        
        with Pool(processes=16) as pool:
            result = pool.map(functions.run_qaoa_dict, args)
        # result = functions.run_qaoa_dict(args[0])
        
        # if 'result' in locals():
        # filename = f', method={method}, degree={degree}, order={order}, depth={depth}, step_size={step_size}, localTerm={localTerm}.npy'
        filename =   f'method={method}, degree={degree}, order={order}, depth={depth}, step_size={step_size}, localTerm={localTerm}'
        # now = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        np.save(path + '/opt_results,     ' + filename + '.npy', result)
        