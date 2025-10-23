from typing import Optional, Any, List, Tuple
import numpy as np
from abc import abstractmethod

class BaseNode:
    """
        This is the implementation of a single neuron node in the computational graph.
        Each node does a memoization of its input and output to aid in the backprop.
    """

    def __init__(self, parents : Optional[List] = None, no_childern : int = 1) -> None:
        self.set_parents(parents or [])
        self.childern : List[List[Tuple['BaseNode', int]]] = [[]] * no_childern  # List[List[Tuple]]
        
        # Used for memoization
        self.x : np.ndarray    = np.array([])   # x is input
        self.y : np.ndarray    = np.array([])   # y is output
        self.dJdy : np.ndarray = np.array([])
        self.dJdx : np.ndarray = np.array([])

        # Flags for memoization
        self.output_present : bool = True
        self.grad_present : bool   = True
    
    def set_parents(self, parents : list) -> None:
        self.parents : List[Tuple['BaseNode', int]] = []
        for index_cur_input, (parent, index_parent_output) in enumerate(parents):
            # index_cur_input is the index of the current node's input array which corresponds
            # to the parent
            # parent is the previous parent node
            # index_parent_output is the index of the parent output array
            # corresponds to the current node
            self.parents.append((parent, index_parent_output))
            parent.add_child(self, index_cur_input, index_parent_output)
    
    def add_child(self, index_output : int, child : 'BaseNode', index_child_input : int) -> None:
        """
            Adds a child of the current node
            :param index_output      : index of the output array to which the child corresponds to
            :param child             : child node
            :param index_child_output: index of the input array of the corresponding child
        """

        self.childern[index_output].append((child, index_child_input))
    
    def get_output(self, index_cur_output : int) -> Optional[float]:
        """
            Used for Forward Propagation
            Gets the output value from the given index out of the current node's output array
            :param index_cur_output: Index to read from the output array
        """

        if self.output_present: # parents have already their outputs ready
            self.x = np.array([
                parent.get_output(index_parent_output)
                for (parent, index_parent_output) in self.parents
            ])
            self.y = self.compute_output()
            self.output_present = False
        return self.y[index_cur_output]
    
    @abstractmethod
    def compute_output(self) -> np.ndarray:
        raise NotImplementedError()
    
    def get_gradient(self, index_cur_input : int):
        """
            Used for Backward Propagation
        """

        if sum(len(child) for child in self.childern) == 0:  # node has no childern
            return np.zeros(self.x[index_cur_input].shape)
        
        if self.grad_present:  # childern already have their grads ready
            self.dJdy = np.array([
                sum(child.get_gradient(index_child_input) for child, index_child_input in childern)
                for childern in self.childern
            ])
            self.dJdx = self.compute_gradient()  # use to compute dydx into dJdx
            self.grad_present = False
        return self.dJdx[index_cur_input]
    
    @abstractmethod
    def compute_gradient(self) -> np.ndarray:
        raise NotImplementedError
    
    def reset_memoization(self):
        self.output_present = True
        self.grad_present = True

class InputNode(BaseNode):
    """
        Specifies the input to the computational graph
        It is the node with no in_degrees
    """

    def __init__(self, value : Optional[float] = None, parents: List | None = None, no_childern: int = 1) -> None:
        """
            :param value: 
        """

        super().__init__(parents, no_childern)
        self.set_value(value)

    def get_output(self, index_cur_output: int) -> Optional[float]:
        """
            The node will simply return its input value back
            :param index_cur_output: 
        """

        return self.value
    
    def set_value(self, value : Optional[float]):
        """
            To set the inputs
            :param value: 
        """

        self.value = value
    
    def compute_gradient(self):
        """

        """

        return np.array([self.dJdy[0]])

class ParameterNode(BaseNode):
    """
        Specifies the parameter values in the graph as either weight or a bias
    """

    def __init__(self, w : np.ndarray, parents: List | None = None, no_childern: int = 1) -> None:
        """
            :param w: parameter to store
        """

        super().__init__(parents, no_childern)
        self.w = w
    
    def compute_output(self):
        """
            Returns the parameter value
        """

        return np.array([self.w])
    
    def compute_gradient(self):
        """
            Returns the gradient with respect to w
        """

        return np.array([self.dJdy[0]])

class SigmoidNode(BaseNode):
    def compute_output(self) -> np.ndarray:
        """
            Returns the sigmoid function evaluation of the input
        """

        return np.array([1 / 1 + np.exp(-self.x[0])])
    
    def compute_gradient(self) -> np.ndarray:
        """
            Returns the gradient of the sigmoid function
        """

        return np.array([self.dJdy[0] * self.y[0] * (1 - self.y[0])])

class GradientNode(BaseNode):
    """
        Acts as the final node of the computational graph, as a child of the cost function node
        It has no out_degrees
    """

    def __init__(self, value : float = 1, parents: List | None = None, no_childern: int = 1) -> None:
        """
            :param value: initialized to 1
        """

        super().__init__(parents, no_childern)
        self.set_value(value)
    
    def set_value(self, value : float):
        """
            Set a gradient value as a init
            :param value
        """

        self.value = value
    
    def compute_output(self) -> np.ndarray:
        """
            Return its input as it performs no operation
        """

        return self.x

    def compute_gradient(self) -> np.ndarray:
        """
            Return the init grad value
        """

        return np.array(self.value)

class AddBiasNode(BaseNode):
    def compute_gradient(self) -> np.ndarray:
        return np.array([self.dJdy[0][:,1:]])
    
    def compute_output(self) -> np.ndarray:
        return np.array([
            np.concatenate((
                    np.ones((self.x[0].shape[0], 1)),
                    self.x[0]
                ),
                axis=1
        )])

class MultiplicationNode(BaseNode):
    def compute_gradient(self) -> np.ndarray:
        return np.array([np.dot(self.dJdy[0], self.x[1].T), np.dot(self.x[0].T, self.dJdy[0])])

    def compute_output(self) -> np.ndarray:
        return np.array([np.dot(self.x[0], self.x[1])])

class TanhNode(BaseNode):
    def compute_gradient(self) -> np.ndarray:
        return np.array([self.dJdy[0] * (1-np.square(self.y[0]))])

    def compute_output(self) -> np.ndarray:
        return np.array([self.dJdy[0] * (1-np.square(self.y[0]))])

class BinaryCrossEntropyNode(BaseNode):
    def compute_gradient(self) -> np.ndarray:
        return np.array([-self.dJdy[0]*(np.log(self.x[1]/(1-self.x[1]))), \
            -self.dJdy[0]*(self.x[0]/self.x[1]-(1-self.x[0])/(1-self.x[1]))])

    def compute_output(self) -> np.ndarray:
        return np.array([-np.sum((self.x[0]*np.log(self.x[1]) + (1-self.x[0])*np.log(1-self.x[1])))])
