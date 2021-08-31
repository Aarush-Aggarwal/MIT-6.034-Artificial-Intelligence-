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
