import numpy as np
from typing import List, Optional, Any
from Node import *

class ComputationalGraph:
    """
        A DAG that encompasses all actions of a neural network
        Contains six types of nodes based on their operations:
        1. Input node       - nodes that store the input (roots of the graphs)
        2. Parameter node   - nodes that modify the inputs
        3. Output node      - nodes that store the prediction of the network
        4. True Output node - nodes that store the true output
        5. Cost node        - nodes that store the cost function between true output and prediction
        6. Gradient node    - node  that initializes the gradient
    """

    def __init__(self, input_nodes : List[InputNode], parameter_nodes : List[ParameterNode],
                 output_nodes : List[Node], true_output_nodes : List[InputNode],
                 cost_node : Node) -> None:
        
        self.input_nodes        = input_nodes
        self.parameter_nodes    = parameter_nodes
        self.output_nodes       = output_nodes
        self.true_output_nodes  = true_output_nodes
        self.cost_node          = cost_node
        self.gradient_node      = GradientNode(parents=[(self.cost_node, 0)])

        self.nodes : List[Node] = self.input_nodes + self.parameter_nodes + \
                                  self.output_nodes + self.true_output_nodes
        
    def forward(self, X : np.ndarray) -> np.ndarray:
        """
            Forward pass through the network
            :param X: input vector
        """

        # reset the grad and output flags so as prepare to compute them again
        self.reset_memoization()

        # init the input nodes with fresh set of input vectors
        for x, node in zip(X, self.input_nodes):
            node.set_value(x)
        
        # propagate through the network (by recursive back calling)
        return np.array([node.get_output(0) for node in self.output_nodes])

    def reset_memoization(self):
        """
            Clear the grad and output caches during forward pass
        """

        for node in self.nodes:
            node.reset_memoization()
        
    def backward(self, Y : np.ndarray) -> Optional[float]:
        """
            Perform a backward propagation through the network
            :param Y: true output vector
        """

        # initialize the true output nodes with a fresh set of true output vectors
        for y, node in zip(Y, self.true_output_nodes):
            node.set_value(y)
        
        # calculate the cost value
        cost = self.cost_node.get_output(0)
        
        # Get gradients of each parameter nodes by backpropagation
        for node in self.parameter_nodes:
            node.get_gradient(0)
        
        return cost
    
    def get_parameter_nodes(self) -> List[ParameterNode]:
        """
            Returns the parameter nodes of the graph
        """

        return self.parameter_nodes
