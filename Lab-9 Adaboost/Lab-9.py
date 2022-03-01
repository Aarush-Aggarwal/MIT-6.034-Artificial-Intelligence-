from math import log
from utils import *

##########################
# Part 1: Helper functions
##########################

# Initialize weights
def initialize_weights(training_points):
    point_to_weight_dict = {}
    for point in training_points:
        point_to_weight_dict[point] = 1/len(training_points)
    return point_to_weight_dict

# Calculate error rates
def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    classifier_to_error_dict = {}
    for classifier, points in classifier_to_misclassified.items():
        error_rate = 0
        for point in points:
            error_rate += point_to_weight[point] 
        classifier_to_error_dict[classifier] = error_rate
        
    return classifier_to_error_dict
    
# Pick the best weak classifier
def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    best_classifier = min(classifier_to_error_rate.keys(), key=classifier_to_error_rate.get)
    
    if classifier_to_error_rate[best_classifier] == 1/2:
        raise NoGoodClassifiersError()
    else:
        return best_classifier
    
# Calculate voting power
def calculate_voting_power(error_rate):
    if error_rate == 0:
        return INF
    elif error_rate == 1:
        return -INF
    
    return make_fraction(1,2) * ln(make_fraction((1-error_rate),error_rate))


def classify_point(point, classifier, classifier_to_misclassified):
    if point in classifier_to_misclassified[classifier]:
        return 1
    else:
        return -1

# Is H good enough?
def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    misclassified = set()
    for point in training_points:
        classification = sum([alpha * classify_point(point, classifier, classifier_to_misclassified) for classifier, alpha in H ])
    
        if classification >= 0:
            misclassified.add(point)
            
    return misclassified
        
def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    if len(get_overall_misclassifications(H, training_points, classifier_to_misclassified)) > mistake_tolerance:
        return False
    else:
        return True
    
# Update weights
def update_weights(point_to_weight, misclassified_points, error_rate):
    for point, old_weight in point_to_weight.items():
        if point in misclassified_points:
            point_to_weight[point] = make_fraction(1,2) * make_fraction(1, error_rate*(old_weight))
        else:
            point_to_weight[point] = make_fraction(1,2) * make_fraction(1, (1-error_rate)*(old_weight))
    
    return point_to_weight
    
    
##################
# Part 2: Adaboost
##################

def adaboost(training_points, classifier_to_misclassified, use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    H = [()]
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        
        point_to_weight_dict = initialize_weights(training_points)
        classifier_to_error_dict = calculate_error_rates(point_to_weight_dict, classifier_to_misclassified)
        
        try:
            best_classifier = pick_best_classifier(classifier_to_error_dict, use_smallest_error)
        except:
            return H
        
        error_rate = classifier_to_error_dict[best_classifier]
        
        alpha = calculate_voting_power(error_rate)
        
        H.append((best_classifier, alpha))
        
        misclassified_points = get_overall_misclassifications(H, training_points, classifier_to_misclassified)
        
        if is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance):
            return H
        
        point_to_weight_dict = update_weights(point_to_weight_dict, misclassified_points, error_rate)
    else:
        print("max rounds exceeded")
    return H
