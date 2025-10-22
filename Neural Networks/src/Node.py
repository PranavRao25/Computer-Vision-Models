from typing import Optional, Any, List, Tuple
import numpy as np
from abc import abstractmethod

class Node:
    """
        This is the implementation of a single neuron node in the computational graph.
        Each node does a memoization of its input and output to aid in the backprop.
    """

    def __init__(self, parents : Optional[List] = None, no_childern : int = 1) -> None:
        self.set_parents(parents or [])
        self.childern : List[List[Tuple['Node', int]]] = [[]] * no_childern  # List[List[Tuple]]

        # Used for memoization
        self.x : List    = None   # x is input
        self.y : List    = None   # y is output
        self.dJdy : List = None
        self.dJdx : List = None

        # Flags for memoization
        self.output_present : bool = True
        self.grad_present : bool   = True
    
    def set_parents(self, parents : list) -> None:
        self.parents : List[Tuple['Node', int]] = []
        for index_cur_input, (parent, index_parent_output) in enumerate(parents):
            # index_cur_input is the index of the current node's input array which corresponds
            # to the parent
            # parent is the previous parent node
            # index_parent_output is the index of the parent output array
            # corresponds to the current node
            self.parents.append((parent, index_parent_output))
            parent.add_child(self, index_cur_input, index_parent_output)
    
    def add_child(self, index_output : int, child : 'Node', index_child_input : int) -> None:
        """
            Adds a child of the current node
            :param index_output      : index of the output array to which the child corresponds to
            :param child             : child node
            :param index_child_output: index of the input array of the corresponding child
        """

        self.childern[index_output].append((child, index_child_input))
    
    def get_output(self, index_cur_output : int) -> float:
        """
            Used for Forward Propagation
            Gets the output value from the given index out of the current node's output array
            :param index_cur_output: Index to read from the output array
        """

        if self.output_present: # parents have already their outputs ready
            self.x = [
                parent.get_output(index_parent_output)
                for (parent, index_parent_output) in self.parents
            ]
            self.y = self.compute_output()
            self.output_present = False
        return self.y[index_cur_output]
    
    @abstractmethod
    def compute_output(self):
        raise NotImplementedError()
    
    def get_gradient(self, index_cur_input):
        """
            Used for Backward Propagation
        """

        if sum(len(child) for child in self.childern) == 0:  # node has no childern
            return np.zeros(self.x[index_cur_input].shape)
        
        if self.grad_present:  # childern already have their grads ready
            self.dJdy = [
                sum(child.get_gradient(index_child_input) for child, index_child_input in childern)
                for childern in self.childern
            ]
            self.dJdx = self.compute_gradient()  # use to compute dydx into dJdx
            self.grad_present = False
        return self.dJdx[index_cur_input]
    
    @abstractmethod
    def compute_gradient(self):
        raise NotImplementedError
    
    def reset_memoization(self):
        self.output_present = True
        self.grad_present = True
