from collections import defaultdict
from nn_problems import *
from math import e
INF = float('inf')
import numpy as np
import matplotlib

#######################
# Part 2: Coding Warmup
#######################

# Exploring different threshold functions
def stairstep(x, threshold=0):
    """ stairstep: Computes the output of the stairstep function using the given threshold (T). """
    return 1 if x >= threshold else 0

    
def sigmoid(x, steepness=1, midpoint=0):
    """ sigmoid: Computes the output of the sigmoid function using the given steepness (S) and midpoint (M) """
    return 1/(1 + e**(-steepness * (x - midpoint)))
    
def ReLU(x):
    """ ReLU: Computes the output of the ReLU (rectified linear unit) function. """
    return max(0, x)

# Measuring performance with the accuracy function
def accuracy(desired_output, actual_output):
    return -0.5 * (desired_output - actual_output)**2


#############################
# Part 3: Forward propagation
#############################

def node_value(node, input_values, neuron_outputs):
    if isinstance(node, str):
        return input_values[node] if node in input_values else neuron_outputs[node]
    return node  # constant input, such as -1

def forward_prop(net, input_values, threshold_fn=stairstep):
    neuron_outputs = {}
    for neuron in net.topological_sort():
        weighted_sum = 0 
        for inc_neighbor in net.get_incoming_neighbors(neuron):
            wire = net.get_wire(inc_neighbor, neuron)
            weighted_sum += node_value(inc_neighbor, input_values, neuron_outputs) * wire.get_weight()
        neuron_outputs[neuron] = threshold_fn(weighted_sum)
    return (neuron_outputs[net.get_output_neuron()], neuron_outputs)


##############################
# Part 4: Backward propagation
##############################

# Gradient ascent

def gradient_ascent_step(func, inputs, step_size):
    from collections import defaultdict
    outputs = defaultdict(list)
    for input1 in inputs:
        for input2 in inputs:
            for input3 in inputs:
                outputs[func(input1, input2, input3)].append([input1, input2, input3])
    max_func, var_list = max(outputs.items())
    return (max_func, var_list)

# Back prop dependencies

def get_back_prop_dependencies(net, wire):
    dependencies = set()
    dependencies.add(wire)
    dependencies.add(wire.startNode)
    dependencies.add(wire.endNode)
    
    for out_neighbor in net.get_outgoing_neighbors(wire.endNode):
        dependencies.add(out_neighbor)
        wires = net._get_wires(wire.endNode,out_neighbor)
        for w in wires:
            dependencies.add(get_back_prop_dependencies(net, w))
    
    return dependencies

