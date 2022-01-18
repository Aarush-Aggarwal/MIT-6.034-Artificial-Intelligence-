from svm_data import *
import math

#####################
# Part 1: Vector Math
#####################

def dot_product(u, v):
    return sum(a*b for a, b in zip(u, v))

def norm(v):
    return math.sqrt(dot_product(v, v))


##########################################
# Part 2: Using the SVM Boundary Equations
##########################################

def positiveness(svm, point):
    return dot_product(svm.w, point) + svm.b

def classify(svm, point):
    if positiveness(svm, point) > 0 :
        return 1
    elif positiveness(svm, point) < 0:
        return -1
    else:
        return 0

def margin_width(svm):
    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    violators = set()
    for point in svm.training_points:
        if point in svm.support_vectors:
            if point.classification != positiveness(svm, point):
                violators.add(point)
        else:
            if abs(positiveness(svm, point)) <= margin_width(svm):
                violators.add(point)
    return violators


########################
# Part 3: Supportiveness
########################

def check_alpha_signs(svm):
    violators = set()
    for point in svm.training_points:
        if point.alpha < 0:
            violators.add(point)
        if point in svm.support_vectors and point.alpha == 0:
            violators.add(point)
        elif point not in svm.support_vectors and point.alpha != 0:
            violators.add(point)
    return violators

def check_alpha_equations(svm):
    summation1 = 0 
    summation2 = [0] * len(svm.training_points[0])
    
    for point in svm.support_vectors:
        summation1 += point.classification * point.alpha
        summation2 = vector_add(scalar_multiply(point.classification*point.alpha, point.coords), summation2)
        
    if summation1 != 0 or svm.w != summation2:
        return False
    else:
        return True
    

#############################
# Part 4: Evaluating Accuracy
#############################

def misclassified_training_points(svm):
    return set( point for point in svm.training_points if classify(svm, point) != point.classification )


###########################################
# Part 5: Training a support vector machine
###########################################

def update_svm_from_alphas(svm):
    b = []
    support_vec = [] 
    for point in svm.training_points:
        if point.alpha > 0:
            support_vec.append(point)
    
    for point in support_vec:
        summation = vector_add(scalar_multiply(point.classification*point.alpha, point.coords), summation)
    
    svm.w = summation
    
    for point in support_vec:
        b.append( point.classification - dot_product(svm.w, point.coords) )
    
    svm.b = (max(b) + min(b)) / 2
    
    svm.support_vectors = support_vec
    
    svm.set_boundary(svm.w, svm.b)
    
    return svm
