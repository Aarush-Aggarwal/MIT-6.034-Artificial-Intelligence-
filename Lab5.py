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
    from collections import defaultdict
    dic = defaultdict(list)
    counter = 0
    for point in data:
        dic[classifier.classify(point)].extend([point])
    return dic

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
    dic = {}
    for point in data:
        classification = target_classifier.classify(point)
        if classification not in dic.keys():
            dic[classification] = 1
        else:
            dic[classification] += 1
    
    disorder = 0.0
    n_b = len(data)
    for n_bc in dic.values():
        disorder += float((n_bc/n_b) * log2(n_bc/n_b))
            
    return(-1.0 * disorder)


def average_test_disorder(data, test_classifier, target_classifier):
    dic = split_on_classifier(data, test_classifier) 
    from collections import defaultdict
    new_dict = defaultdict(list)
    for feature, points in dic.items():
        new_dict[feature].extend([len(points), branch_disorder(points, target_classifier)])
    
    avg_disorder = 0.0
    for v in new_dict.values():
        avg_disorder += (v[0]/len(data)) * v[1]
    
    return(avg_disorder)

##################################
# Part 1D: Constructing an ID Tree
##################################

def find_best_classifier(data, possible_classifiers, target_classifier):
    test_disorders = {}
    for classifier in possible_classifiers:
        test_disorders[classifier] = average_test_disorder(data, classifier, target_classifier)
    
    best_classifier = min(test_disorders, key=test_disorders.get)
    min_dict = split_on_classifier(data, best_classifier)
    if len(min_dict.keys()) == 1:
        raise api.NoGoodClassifiersError("the classifier has only one branch") 
    
    return best_classifier
