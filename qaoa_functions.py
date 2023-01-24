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

        # self.G: Graph with the interaction pairs
        self.G = nx.random_regular_graph(d,nqubits)

        # self.J: Dict that contains all J values of the interactions
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
    
    def hamiltonian_qngd(self):
        """
        - provided by a classmate(Manuel) in order to implement qn_gd
        """
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
    
    def draw_graph(self, with_labels=True, alpha=0.8, node_color='tab:blue', **kwargs):
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
        
    def QAOA(self, p, method='QNGD', max_iterations=50000, conv_tol=1e-25, step_size=0.01, print_step=10, process_number=1):
        dev = qml.device('default.qubit', self.nqubits)        
        
        # Step #1: Make circuit:  
        @qml.qnode(dev, interface="autograd")
        def cost_function(params):
            self.make_circuit(params, p)
            return qml.expval(self.hamiltonian2())
        
        # Info init_params: [beta[0],..., beta[p], gamma[0],..., gamma[p]]
        rng = np.random.default_rng()
        init_params = (rng.random(size=2*p)-0.5)*4*np.pi
        # init_params = np.append(np.linspace(1,1,p), np.linspace(2,2,p))
        print(init_params)
        init_params = np.array(init_params, requires_grad=True)
        
        if method == 'AdamGD':
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
            # Take step
            times = [time.time()]
            params, prev_energy = opt.step_and_cost(cost_function, params)
            times.append(time.time())
            opt_param_history.append(params)
            opt_cost_history.append(prev_energy)
            times.append(time.time())
            energy = cost_function(params)
            times.append(time.time())
            # Calculate difference between new and old energies
            conv = np.abs(energy - prev_energy)

            if n % print_step == 0:
                print(
                    "Process #{:.0f}: Iteration = {:},  Energy = {:.12f} Ha,  Convergence parameter = {"
                    ":.12f} Ha".format(process_number, n, energy, conv)
                )
                print(np.diff(times))

            if ((conv <= conv_tol) and (n >= 20)):
                break
        return dev, cost_function, opt_param_history, opt_cost_history   
    
