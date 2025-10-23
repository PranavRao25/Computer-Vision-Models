import numpy as np
from typing import List, Optional
from abc import abstractmethod
from .Node import *

class OptimizingAlgorithm:
    """
        Abstract class to implement algorithms for improving the parameters of a neural network
        for better performance
    """

    def __init__(self, parameter_nodes : List[ParameterNode], learning_rate : float = 1e-3) -> None:
        """
            :param parameter_nodes: all parameter nodes of the graph
            :param learning_rate  : learning rate of the algorithm
        """

        self.parameter_nodes = parameter_nodes
        self.lr = learning_rate

    def optimize(self, batch_size : int = 1):
        """
            Update the values of the parameters in the direction of the gradient descent

            :param batch_size: used if input is batched
        """

        for i, node in enumerate(self.parameter_nodes):
            step = self.compute_step(i, node.get_gradient(0) / batch_size)
            node.w -= self.lr * step

    @abstractmethod
    def compute_step(self, parameter_index, grad):
        """
            Used to compute a step which reduces the gradient
        """

        raise NotImplementedError

class GradientDescent(OptimizingAlgorithm):
    """
        Implementation of Gradient Descent Algorithm by specifying the step value
    """

    def compute_step(self, parameter_index : int, grad : np.ndarray):
        """
            Return the grad step

            :param parameter_index: which parameter to be updated
            :param grad           : gradient of the parameter
        """

        return grad