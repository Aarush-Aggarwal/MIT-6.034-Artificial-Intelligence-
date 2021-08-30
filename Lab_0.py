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

# For example:
#  depth('x') => 0
#  depth(('expt', 'x', 2)) => 1
#  depth(('+', ('expt', 'x', 2), ('expt', 'y', 2))) => 2
#  depth(('/', ('expt', 'x', 5), ('expt', ('-', ('expt', 'x', 2), 1), ('/', 5, 2)))) => 4

--------------------------------------------------------------------------------------------
# The depth of a list is one more than the maximum depth of its sub-lists.
# map(depth, expr) means apply the function "depth" to each element of expr.

def depth(expr):
    return max(map(depth, expr))+1 if isinstance(expr, tuple) else 0 

print(depth('x'))
print(depth(('expt', 'x', 2)))
print(depth(('+', ('expt', 'x', 2), ('expt', 'y', 2))))
print(depth(('/', ('expt', 'x', 5), ('expt', ('-', ('expt', 'x', 2), 1), ('/', 5, 2)))))
