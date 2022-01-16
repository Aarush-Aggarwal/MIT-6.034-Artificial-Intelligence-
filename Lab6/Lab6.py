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


# Basic back propagation

# Computing δB
def calculate_deltas(net, desired_output, neuron_outputs):
    _dict = {}
    graph = net.topological_sort()
    graph.reverse()
    for neuron in graph:
        if net.is_output_neuron(neuron):
            delta_val = neuron_outputs[neuron] * (1 - neuron_outputs[neuron]) * (desired_output - neuron_outputs[neuron])
        else:
            summation = 0
            for out_neighbor in net.get_outgoing_neighbors(neuron):
                wire = net.get_wire(neuron, out_neighbor)
                summation += ( wire.get_weight() * _dict[neuron] )
                delta_val = neuron_outputs[neuron] * (1 - neuron_outputs[neuron]) * summation
        _dict[neuron] = delta_val     
    
    return _dict

# Updating weights
def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    _dict = calculate_deltas(net, desired_output, neuron_outputs)
    for wire in net._get_wires():
        delta_val = _dict[wire.endNode]
        delta_w = r * node_value[wire.startNode, input_values, neuron_outputs] * delta_val
        new_weight = wire.get_weight() + delta_w
        wire.set_weight(new_weight)
        
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    num_iterations = 0
    calc_output, neuron_outputs = forward_prop(net, input_values, sigmoid)
    while accuracy(desired_output, calc_output) < minimum_accuracy:
        net = update_weights(net, input_values, desired_output, neuron_outputs, r)
        num_iterations += 1
        calc_output, neuron_outputs = forward_prop(net, input_values, sigmoid)
    
    return (net, num_iterations)
