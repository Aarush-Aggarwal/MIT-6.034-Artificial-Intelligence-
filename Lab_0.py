-------------------
# Warm-up stretch #
-------------------

# Count Pattern

# count_pattern(pattern lst), which counts the number of times a certain pattern of symbols appears in a list, including overlaps. 
# So count_pattern( ('a', 'b'), ('a','b', 'c', 'e', 'b', 'a', 'b', 'f')) should return 2, and 
# count_pattern(('a', 'b', 'a'), ('g', 'a', 'b', 'a', 'b', 'a','b', 'a')) should return 3. 

-----------------------------------------------------------------------------------------------------------------------------------

def count_pattern(pattern, lst):
    return sum(pattern == lst[pos:pos+len(pattern)] for pos in range(len(lst)))

print(count_pattern(('a', 'b'), ('a', 'b', 'c', 'e', 'b', 'a', 'b', 'f')))
print(count_pattern(('a', 'b', 'a'), ('g', 'a', 'b', 'a', 'b', 'a', 'b', 'a')))


--------------------
# Expression depth #
--------------------

# expression depth depends on the depths of all of its children (arguments). it is 1 more than the maximum of depths of its children. 
# If you think of the expression as a tree, the expression depth is the height of the tree.
# isinstance determines whether a variable points to a list. it takes two arguments: the variable to test, and the type (or tuple of types) to compare it to.
# map(depth, expr) means apply the function "depth" to each element of expr.

# For example:
#  depth('x') => 0
#  depth(('expt', 'x', 2)) => 1
#  depth(('+', ('expt', 'x', 2), ('expt', 'y', 2))) => 2
#  depth(('/', ('expt', 'x', 5), ('expt', ('-', ('expt', 'x', 2), 1), ('/', 5, 2)))) => 4

--------------------------------------------------------------------------------------------

def depth(expr):
    return max(map(depth, expr))+1 if isinstance(expr, tuple) else 0 

print(depth('x'))
print(depth(('expt', 'x', 2)))
print(depth(('+', ('expt', 'x', 2), ('expt', 'y', 2))))
print(depth(('/', ('expt', 'x', 5), ('expt', ('-', ('expt', 'x', 2), 1), ('/', 5, 2)))))


------------------
# Tree reference #
------------------

# tree_ref procedure will take a tree and an index, and return the part of the tree (a leaf or a subtree) at that index.
# for trees, indices will have to be lists of integers.

--------------------------------------------------

def tree_ref(tree, index):
        return tree_ref(tree[index[0]], index[1:]) if index else tree

tree = (((1, 2), 3), (4, (5, 6)), 7, (8, 9, 10))

print(tree_ref(tree, (3, 1)))
print(tree_ref(tree, (1, 1, 1)))
print(tree_ref(tree, (0,)))


--------------------
# Symbolic algebra #
--------------------

# algebra.py. 
# do_multiply

-------------------------------------------------------------------------

def do_multiply(expr1, expr2):
    """
    You have two Expressions, and you need to make a simplified expression
    representing their product. They are guaranteed to be of type Expression
    -- that is, either Sums or Products -- by the multiply() function that
    calls this one.

    So, you have four cases to deal with:
    * expr1 is a Sum, and expr2 is a Sum
    * expr1 is a Sum, and expr2 is a Product
    * expr1 is a Product, and expr2 is a Sum
    * expr1 is a Product, and expr2 is a Product

    You need to create Sums or Products that represent what you get by
    applying the algebraic rules of multiplication to these expressions,
    and simplifying.

    Look above for details on the Sum and Product classes. The Python operator
    '*' will not help you.
    """
result = []
    if isinstance(expr1,Sum) and isinstance(expr2,Sum):
        for elem1 in expr1:
            for elem2 in expr2:
                result.append(Product([elem1,elem2]).simplify())
        return Sum(result).simplify()
    
    elif isinstance(expr1,Sum) and isinstance(expr2,Product):
        for elem in expr1:
            result.append(Product([elem,expr2]).simplify())
        return Sum(result).simplify()
    
    elif isinstance(expr1,Product) and isinstance(expr2,Sum):
        for elem in expr2:
            result.append(Product([elem,expr1]).simplify())
        return Sum(result).simplify()
    
    elif isinstance(expr1,Product) and isinstance(expr2,Product):
        for elem in expr1:
            result.append(elem)
        for elem in expr2:
            result.append(elem)
        return Product(result)

    
-----------------------
# Built-in data types #
-----------------------    

# remove_from_string that takes in 1) a string to copy, and 2) a string of letters to be removed. 

-----------------------------------------------------------

def remove_from_string(string, letters):
    return ''.join([c for c in string if c not in letters])

print(remove_from_string("6.034", "46"))


# tally_letters takes in a string of lowercase letters and returns a dictionary mapping each letter to the number of times it occurs in the string

--------------------------------------------------------------------------------------------------------------------------------------------------

def tally_letters(string):
    return { key: string.count(key) for k in string }

print(tally_letters("hello"))
