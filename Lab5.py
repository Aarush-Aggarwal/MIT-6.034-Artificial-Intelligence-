import api
import data
import math

log2 = lambda x: math.log(x, 2)
INF = float('inf')

######################################################
# Part 1A: Using an ID Tree to classify unknown points
######################################################

def id_tree_classify_point(point, id_tree):
    if id_tree.is_leaf():
        id_tree.get_node_classification()
    
    return id_tree_classify_point(point, id_tree.apply_classifier(point))

###########################################
# Part 1B: Splitting Data with a Classifier
###########################################

def split_on_classifier(data, classifier):
    dic = {}
    counter = 0
    for point in data:
        dic[classifier.classify(point)] = [] + [point]
    return dic

split_on_classifier(data.angel_data, api.feature_test("Shape"))

###############################
# Part 1C: Calculating Disorder
###############################

ball1 = {"size": "big", "color": "brown", "type": "basketball"}
ball2 = {"size": "big", "color": "white", "type": "soccer"}
ball3 = {"size": "small", "color": "white", "type": "lacrosse"}
ball4 = {"size": "small", "color": "blue", "type": "lacrosse"}
ball5 = {"size": "small", "color": "yellow", "type": "tennis"}
ball_data = [ball1, ball2, ball3, ball4, ball5] 

ball_type_classifier = api.feature_test("type")

def branch_disorder(data, target_classifier):
    from collections import Counter
    values = []
    n_b = len(data)
    
    for datum in data:
        if target_classifier in datum.keys():
            values.append(datum[target_classifier])
        else:
            raise api.ClassifierError("Target Classifier not in Data Set")
        
    for n_bc in Counter(values).values:
        disorder = float(disorder + n_bc/n_b * log2(n_bc/n_b))
            
    return -1.0 * disorder

branch_disorder([ball3, ball4, ball5], ball_type_classifier)

def average_test_disorder(data, test_classifier, target_classifier):
    pass

##################################
# Part 1D: Constructing an ID Tree
##################################

# def find_best_classifier(data, possible_classifiers, target_classifier):
