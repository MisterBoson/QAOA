"""

Author:     @mchadolias
"""

# Packages
import matplotlib.pyplot as plt
from pennylane import numpy as np
import networkx as nx
import qaoa_functions as fn
import datetime
from multiprocessing import Pool
import os
#import pandas as pd
#import seaborn as sns
plt.style.use('classic')


def valid(param, n_repititions):
    """
    Check if the initial values of the parameters are valid. In order to prevent errors in the next loop.
    """
    if np.isscalar(param):
        return np.repeat(param, n_repititions)
    elif len(param)== n_repititions:
        return np.array(param)
    else:
        raise ValueError('Shape of param does not match requirements')


if __name__ == '__main__':
    path_wd = r'/home/mrboson/Documents/Academic_Route/MS_Physics/Courses/Computational Physics /Code_new_repo'
    while not (os.getcwd()==path_wd):
        os.chdir(path_wd)   
    
            
    # Parameters    
    #method_rep    = ['QNGD', 'GD', 'AdamGD']
    method_rep    = ['GD','GD','GD','GD','GD','GD',
                     'QNGD','QNGD','QNGD','QNGD','QNGD','QNGD',
                     'AdamGD','AdamGD','AdamGD','AdamGD','AdamGD','AdamGD']
    #method_rep    = ['GD','GD',"GD"]
    order_rep     = [5,6,7,8,9,10,
                     5,6,7,8,9,10,
                     5,6,7,8,9,10]   # Number of Qubits
    degree_rep    = [i-1 for i in order_rep]  # Number of connected nodes
    depth_rep     = 5
    #step_size_rep = [0.1,0.01,0.01]
    step_size_rep = [0.001,0.001,0.001,0.001,0.001,0.001,
                     0.01, 0.01 ,0.01 ,0.01 ,0.01, 0.01, 
                     0.001,  0.001 , 0.001 , 0.001 , 0.001,0.001]
    localTerm_rep = True
    random_graph_rep = True
    #max_iterations= 10    #to quickly verify my code
   
    
    parent_path = path_wd + r'/Results/'
    n_repititions = len(method_rep) 
    n_parallel = 8
    r_value, r_mean, r_std= [], np.zeros(n_repititions), np.zeros(n_repititions)
    
    method_rep = valid(method_rep, n_repititions)
    degree_rep = valid(degree_rep, n_repititions)
    order_rep = valid(order_rep, n_repititions)
    depth_rep = valid(depth_rep, n_repititions)
    step_size_rep = valid(step_size_rep, n_repititions)
    localTerm_rep = valid(localTerm_rep, n_repititions)
    random_graph_rep = valid(random_graph_rep, n_repititions)
    
    
    for i in range(n_repititions):
        method = method_rep[i].item()
        degree = degree_rep[i].item()
        order = order_rep[i].item()
        depth = depth_rep[i].item()
        step_size = step_size_rep[i].item()
        localTerm = localTerm_rep[i].item()
        random_graph = random_graph_rep[i].item()
        print('method: ', method)
        print("number_of_qubits", order )
        print('depth: ', depth)
        identification = '-New_Batch-'
        path = parent_path + identification + datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        os.makedirs(path)
        args = []
        for k in range(n_parallel):
            args_i = {'process_number':  k, 
                      'path':            path,
                      'method':          method, 
                      'd':               degree, 
                      'nqubits':         order, 
                      'p':               depth, 
                      'localTerm':       localTerm,
                      'step_size':       step_size,
                      #'max_iterations':  max_iterations,
                      }
            args.append(args_i)
        
        
        with Pool(processes=16) as pool:
            result = pool.map(fn.run_qaoa_dict, args)
          
        #r-value
        dataset_r_value= [ result[k]["opt_cost_history"][-1] for k in range(n_parallel)]
        min_eigenvalue= result[0]['minimum_eigenvalue']
        r_value.append(fn.approximate_ratio(min_eigenvalue ,dataset_r_value))
        r_value[i], r_mean[i] , r_std[i] = fn.approximate_ratio(result[0]['minimum_eigenvalue'] ,dataset_r_value)
                       
        
        #plotting
        path_fig= path+r'/Figures'
        os.makedirs(path_fig)
        title= "Method:{} Step_size:{}".format(method,step_size)
        plt_name= "method_{}_step_size_{}_order_{}_degree_{}_depth_{}.pdf".format(method,step_size,order,degree,depth)
        #plt.show() #for the nx graph
        
        plt.figure()
        for j in range(n_parallel):
            x_data= np.arange(0,result[j]["number_of_iterations"]+1)
            y_data= result[j]["opt_cost_history"]
            plt.plot(x_data,y_data, label="sample: #{}:".format(result[j]["process_number"]))
        plt.axhline(y=result[0]['minimum_eigenvalue'], linestyle='--', color='black')
        #plt.xlim(right=250, left=0)
        plt.ylim(bottom= result[0]['minimum_eigenvalue']-1 )
        plt.grid()
        plt.legend(prop={'size': 7.5})
        plt.title(title)
        plt.ylabel("Cost Function")
        plt.xlabel("#Iterations")
        plt.savefig(os.path.join(path_fig, plt_name))
        plt.show()
        
        filename =   f'method={method}, degree={degree}, order={order}, depth={depth}, step_size={step_size}, localTerm={localTerm}'
        np.save(path + '/opt_results,     ' + filename + '.npy', result) #save data of each run 

#plot r-value
plt_name= "approximate-ratio-as-a-function-of-order-of-system.pdf"
x_data= np.linspace(0,1,n_repititions)
ax= plt.axes()
for i in range(n_repititions):
    ax.errorbar(x_data[i]+0.1, r_mean[i], yerr= r_std[i], marker="o", ls='', label= 'method-{}-order-{}-step_size-{}'.format(method_rep[i].item(),order_rep[i].item(),step_size_rep[i].item()))
plt.xlabel("Different Optimizers")
plt.ylabel("r-value")
ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.xlim(right= 1.3 , left= 0)
plt.legend(loc=4 , prop={'size': 6.5})
plt.title('Approximate Ratio for the optimizers Vanilla GD, AdamGD and QNGD')
plt.savefig(os.path.join(path_fig, plt_name))
plt.show()
