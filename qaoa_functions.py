"""

Author:     @mchadolias
"""

"""
Intended to by used in conjuction with qaoa_main. 
This script works as a library for the main class (Random_Ising) provided
"""

#Packages
from pennylane import numpy as np
import pennylane as qml
from pennylane import qaoa
import networkx as nx

"""
Code is based by the following sources:
# Code by Classmate Manuel
# Code provided by supervisor Naeimeh
#https://discuss.pennylane.ai/t/comparing-quantum-natural-gradient-with-adam/2129/4

"""

    
class RandomIsing:
    def __init__(self, d, nqubits, localTerm=True):
        """
        Initialization of Hamiltionian consisting of pair and local terms. The constants correspond to:
            d: connectivity of the graph
            nqubits: number of qubits
            localTerm: bool parameter of whether local terms exist in the hamiltionian
        """
        self.d         = d
        self.nqubits   = nqubits
        self.localTerm = localTerm

        self.G = nx.random_regular_graph(d,nqubits)
        self.J = {pair: 2*np.random.randint(2)-1 for pair in list(self.G.edges())}

        if localTerm:
            self.h = 2*np.random.randint(2, size = nqubits)-1
        else:
            self.h = np.zeros(nqubits)


    def set_coupling(self, J_new):
        if np.isscalar(J_new):
            J_new = np.repeat(J_new, len(self.J))
        keys = list(self.J.keys())
        for i in range(len(self.J)):
            self.J[keys[i]] = J_new[i]

    def set_local_weights(self, h_new):
        if np.isscalar(h_new):
            h_new = np.repeat(h_new, self.nqubits)
        self.h = h_new

    def hamiltonian(self):
        coeffs = np.append([self.J[pair] for pair in list(self.G.edges())], self.h)
        operators = [qml.PauliZ(x1) @ qml.PauliZ(x2) for x1,x2 in list(self.G.edges())]
        operators += [qml.PauliZ(x) for x in range(self.nqubits)]
        return qml.Hamiltonian(coeffs, operators)
    
    def hamiltonian_qng(self):
        operator_list = [self.J[(x1,x2)]*qml.PauliZ(x1) @ qml.PauliZ(x2) for x1,x2 in list(self.G.edges())]
        operator_list += [self.h[x]*qml.PauliZ(x) for x in range(1,self.nqubits)]
        operator = self.h[0]*qml.PauliZ(0)
        for op in operator_list:
            operator += op
        return operator
    
    def mixer(self):
        coeffs = np.ones(self.nqubits)
        operators = [qml.PauliX(x) for x in range(self.nqubits)]
        return qml.Hamiltonian(coeffs, operators)
    
    def qaoa_layer(self, alpha, gamma):
        qaoa.cost_layer(gamma, self.hamiltonian())
        qaoa.mixer_layer(alpha, self.mixer())
    
    def make_circuit(self, params, p):
        for i in range(self.nqubits):
            qml.Hadamard(i)
        qml.layer(self.qaoa_layer, p, params[:p], params[p:])#, hamilton=hamilton, mixer=mixer)
    
    def draw_graph(self, with_labels=True, alpha=0.8, node_color='tab:orange', **kwargs):
        print(kwargs)
        nx.draw(self.G, with_labels=with_labels, alpha=alpha, node_color=node_color, **kwargs)
        return kwargs

    def draw_graph_weights(self,
                           node_color='tab:orange',
                           font_family="sans-serif",
                           font_size=6,
                           **kwargs):
        def kw(kwargs, kwargs_keys):
            return {key: kwargs[key] for key in kwargs if key in kwargs_keys}

        keys = []
        pos = nx.spring_layout(self.G, **kw(kwargs,keys))  # positions for all nodes - seed for reproducibility
        keys = ['node_size', 'node_shape', 'alpha']
        nx.draw_networkx_nodes(self.G, pos, node_color=node_color, **kw(kwargs,keys))
        keys = ['width']
        nx.draw_networkx_edges(self.G, pos, **kw(kwargs,keys))
        keys = []
        labels = {n: f'{n} ({self.h[n]:.0f})' for n in self.G}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=font_size,
                                font_family=font_family, **kw(kwargs,keys))
        keys = []
        nx.draw_networkx_edge_labels(self.G, pos, self.J, **kw(kwargs,keys))
    
        
    def QAOA(self, p, method='QNGD', max_iterations=2500, conv_tol=1e-15, step_size=0.01, process_number=1):
        #dev = qml.device('qiskit.aer', self.nqubits, backend='aer_simulator_statevector')
        dev = qml.device('default.qubit', self.nqubits)
            
        @qml.qnode(dev, interface="autograd")
        def cost_function(params):
            self.make_circuit(params, p)
            return qml.expval(self.hamiltonian_qng())
        
        rng = np.random.default_rng()
        init_params = (rng.random(size=2*p)-0.5)*4*np.pi #initialization of the initial parameters of the optimizer
        init_params = np.array(init_params, requires_grad=True)
        
        if method == 'GD':
            # Gradient descent
            opt = qml.GradientDescentOptimizer(stepsize=step_size)
        elif method == 'AdamGD':
            # Gradient descent (Adam)
            opt = qml.AdamOptimizer(stepsize=step_size)
        elif method == 'QNGD':
            # Quantum natural gradient descent
            opt = qml.QNGOptimizer(stepsize=step_size, approx="block-diag")
        else:
            raise ValueError(f"method must be 'GD', 'QNGD' or 'AdamGD', got '{method}' ")
        
        params = init_params
        opt_param_history = [params]
        opt_cost_history = []

        print(f'Initial params: {init_params}\nInitial Energy: {cost_function(init_params)}\n')
        for n in range(max_iterations):
            params, prev_energy = opt.step_and_cost(cost_function, params)
            opt_param_history.append(params)
            opt_cost_history.append(prev_energy)
            energy = cost_function(params)
            conv = np.abs(energy - prev_energy) #calculation of the tolerance 

            if n % 50 == 0:
                print(
                    "Process #{:.0f}: Iteration = {:},  Energy = {:.12f} Ha,  Convergence parameter = {"
                    ":.12f} Ha".format(process_number, n, energy, conv))
            
            if (( conv <= conv_tol)): #or (n==50)):
                print(
                    "The optimizer has not been converged after Num_Steps={:} and Convergence parameter = {:.12f}".format(n,conv))
                break
           
        print()
        print("Final value of the energy = {:.8f} Ha".format(energy))
        print("Number of iterations = ", n)
        return dev, cost_function, opt_param_history, opt_cost_history, n  
    
def run_qaoa(process_number, path,
             d, nqubits, localTerm, p, # Arguments for Initialization
             G=None, J=None, h=None,   # Graph and weights
             method='QNGD', max_iterations=2500, conv_tol=1e-20, step_size=0.01):
    """
    Main function that runs the QAOA problem.
    
    Parameters
    ----------
    process_number : TYPE
        The different samples that calculate in parallel the same pipeline.
    path : TYPE
        The path to save the results.
    d : TYPE
        Connectivity of the system.
    nqubits : TYPE
        Order of the system. The number of qubits of the circuit.
    localTerm : TYPE
        Check if our Hamiltonian has local terms, which means indivindual gates of the circuit.
    p : TYPE
        Depth of the optimizer. The number of times that the Hamiltonian is applied to the circuit.
       
    # Graph and weights
    G : TYPE, optional
    J : TYPE, optional
    h : TYPE, optional
        # In case the hamiltonian was loaded and not randomly generated.
             
    method : TYPE, optional
        Define the type of the optimizer for the run. The default is 'QNGD'.
    max_iterations : TYPE, optional
        Maximum number of iterations for the optimizer. The default is 2500.
    conv_tol : TYPE, optional
        Convolution tolerance. The default is 1e-20.
    step_size : TYPE, optional
        Step size of the optimizer. The default is 0.01.

    Returns
    -------
    result : TYPE
        A dictionary with the desired parameters.
    """
    mod = RandomIsing(d, nqubits, localTerm)
    if G:
         mod.J = J
         mod.h = h
    hamiltonian = qml.matrix(mod.hamiltonian_qng())
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    print('Process #{:.0f}: smallest eigenvalue: '.format(process_number), np.min(eigenvalues))
    dev, cost_function, opt_param_history, opt_cost_history, number_iterations = mod.QAOA(p, method=method, step_size=step_size, max_iterations=max_iterations, process_number=process_number)
    
    result = {'device':             dev,
              'process_number':     process_number,
              'opt_param_history':  opt_param_history,
              'opt_cost_history':   opt_cost_history,
              'eigenvalues':        eigenvalues,
              'minimum_eigenvalue': np.min(eigenvalues).real,
              'eigenvectors':       eigenvectors,
              'method':             method,
              'degree':             d,
              'depth':              p,
              'order':              nqubits,
              'step_size':          step_size,
              'conv_tol':           conv_tol,
              'max_iterations':     max_iterations,
              "number_of_iterations": number_iterations,
              #'localTerm':          localTerm,
              #'J':                  mod.J,
              #'h':                  mod.h,
              }
    filename = f'method={method}, degree={d}, order={nqubits}, depth={p}, step_size={step_size}, localTerm={localTerm}, process_number={process_number}'
    np.save(path + '/single_results, ' + filename + '.npy', result)
    return result 

def approximate_ratio(eigenvalue,  opt_value):
    """
    Calculation of the approximate ration of each optimizer run.

    Parameters
    ----------
    eigenvalue : TYPE
        The real minimum eigenvalue of the hamiltian system.
    opt_value : TYPE
        The last energy value calculated by the optimizer.

    Returns
    -------
    r_value : TYPE
        The approximate ratio value of each sample.
    r_mean : TYPE
        The mean approximate ratio. 
    r_std : TYPE
        The variance of the approximate ratio.
    """
    r_value= opt_value/eigenvalue
    r_mean= np.mean(r_value)
    r_std= np.std(r_value, ddof=1)
    if r_value.any() > 1 or r_mean > 1 or r_std > 1:
        print('There is an error in the calculation...')
    return r_value , r_mean, r_std
    

def run_qaoa_dict(args):
    return run_qaoa(**args)






