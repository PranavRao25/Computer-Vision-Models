class Node:
    """
        This is the implementation of a single neuron node in the computational graph.
        Each node does a memoization of its input and output to aid in the backprop.
    """

    def __init__(self, parents = None, no_childern = 1) -> None:
        self.set_parents(parents or [])
        self.childern = [[]] * no_childern

        # Used for memoization
        self.x = None   # x is input
        self.y = None   # y is output
        self.dJdy = None
        self.dJdx = None

        # Flags for error
        self.output_dirty = True
        self.grad_dirty   = True
    
    def set_parents(self, parents):
        self.parents = []
        for cur_input, (parent, parent_output) in enumerate(parents):
            self.parents.append((parent, parent_output))
            parent.add_child(self, cur_input, parent_output)
    
    def add_child(self, child, child_input, cur_output):
        self.childern[cur_output].append((child, child_input))
        