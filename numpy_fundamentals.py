"""
NumPy Fundamentals — Complete Reference Script
================================================
A single, runnable file covering core NumPy concepts, with a comment
explaining nearly every line. Organized by topic, from basics to
real-world practical examples.

Author: Ahmad
"""

import numpy as np
import pandas as pd

# =========================================================================
# 1. WHAT IS NUMPY AND WHY IT MATTERS
# =========================================================================
# NumPy ("Numerical Python") converts data into numbers for preprocessing
# -- Machine Learning models can only work with numbers.
# Key ideas: "broadcasting" (element-wise math without manual loops), and
# the "array" (an efficient arrangement of data). Pandas itself is built
# on top of NumPy.


# =========================================================================
# 2. CREATING ARRAYS AND CORE PROPERTIES
# =========================================================================

a1d = np.array([1, 2, 3])                                        # 1D array (a vector)
print(a1d)
print(type(a1d))                                                   # <class 'numpy.ndarray'>

a2d = np.array([[1, 2, 3], [4, 5, 6]])                              # 2D array (a matrix)
print(a2d)

a3d = np.array([
    [[1, 2, 3.1], [4, 5, 6], [7, 8, 9]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
])                                                                     # 3D array (a "tensor")
print(a3d)

print(a3d.shape)     # (2, 3, 3)   -- size along each dimension
print(a1d.shape)      # (3,)
print(a2d.shape)       # (2, 3)
print(a3d.ndim)          # 3 -> number of dimensions
print(a3d.dtype)          # float64 -- because one value (3.1) is a decimal
print(a2d.dtype)           # int64 -- all whole numbers
print(a3d.size)              # total individual elements (2*3*3 = 18)
print(type(a3d))               # numpy.ndarray

# A 2D NumPy array converts directly into a Pandas DataFrame
df = pd.DataFrame(a2d)
print(df)


# =========================================================================
# 3. THREE WAYS TO CREATE ARRAYS: ONES, ZEROS, RANDOM
# =========================================================================

tarray = np.array([1, 2, 3])
print(tarray)

# a) Ones / Zeros -- useful as starting templates or placeholders
ones = np.ones((2, 2))
print(ones)

ones = np.ones((4, 4))
print(ones)
print(ones.dtype)     # float64 by default
print(type(ones))      # numpy.ndarray

zeros = np.zeros((4, 3))
print(zeros)
print(zeros.dtype)    # float64
print(type(zeros))     # numpy.ndarray

# b) arange -- a range of numbers: start, stop (excluded), step
range_array = np.arange(1, 21, 3)
print(range_array)     # we can create a vector of numbers first, then reshape into an array

# c) Random numbers
random_matrix = np.random.randint(0, 10, size=(3, 5))   # random ints 0-9, shape (3,5)
print(random_matrix)
print(random_matrix.shape)     # (3, 5)
print(random_matrix.size)       # 15

rand_matrix = np.random.rand(3, 5)   # random floats between 0 and 1 (auto-ranged, not manual)
print(rand_matrix)
print(rand_matrix.sum())           # sum of every element

# Reproducibility: if you share a project using random numbers, other people
# will get DIFFERENT results each run unless you fix the "seed".
np.random.seed(seed=0)
rand_matrix = np.random.rand(3, 5)     # will always be identical whenever seed=0 is set first
print(rand_matrix)
print(rand_matrix.sum())


# =========================================================================
# 4. ACCESSING ELEMENTS (INDEXING)
# =========================================================================

print(a1d)
print(a2d)
print(a3d)

print(a3d[0])   # first "layer" of the 3D array (a full 2D sub-array)
print(a2d[0])    # first row of the 2D array
print(a2d[1])     # second row
print(a3d[1])      # second layer of the 3D array


# =========================================================================
# 5. MATRIX SLICING
# =========================================================================
# array[dim0_slice, dim1_slice, dim2_slice, ...] -- one slice per dimension

print(a3d[0:2, :1])          # row slicing: outer dim 0-1, inner dim only first row
print(a3d[0:2, :1, :2])        # column slicing: also limit the innermost dimension to first 2
print(a3d[0:2, 0:1, 0:2])       # same as above, written explicitly instead of shorthand ':'

randomm = np.random.randint(10, size=(3, 4, 5))
print(randomm)
print(randomm[0:3:2, 0:2, 0:3])   # "0:3:2" = start:stop:step -> skips every other item
print(randomm.size)                 # total elements = 3*4*5 = 60

randomm4d = np.random.randint(10, size=(2, 3, 4, 5))   # a 4D array
print(randomm4d)
print(randomm4d[:1, :1, :1, :1])    # just the first slice of every dimension


# =========================================================================
# 6. ARRAY MANIPULATION -- ARITHMETIC & BROADCASTING
# =========================================================================
# "Broadcasting" = NumPy applies math element-by-element automatically,
# without you writing a manual loop (one-to-one mapping).

a1dd = np.array([1, 2, 3, 4])
print(a1dd)

print(a1dd + ones[0])    # adds element-wise (sliced 'ones' to match shape for this demo)
print(a1dd - ones[0])
print(a1dd * ones[0])

a3ddd = np.array([
    [[1, 2, 3, 3], [4, 5, 6, 4], [7, 8, 9, 5]],
    [[1, 2, 3, 7], [4, 5, 6, 8], [7, 8, 9, 1]]
])

a2ddd = np.array([
    [1, 2, 3, 4],
    [4, 5, 6, 7]
])

t = a3ddd[:2, :2, :3]
print(t)

v = a3ddd[:2, :2, :3]
print(v)

print(t + v)   # element-wise addition
print(t * v)    # element-wise multiplication

print(a1dd / ones[0])       # element-wise division
print(a1dd / a2ddd)           # broadcasting a (4,) array against a (2,4) array
print(a1dd // a2ddd)            # floor division -- rounds each result down
print(a1dd ** a2ddd)              # element-wise exponentiation (power)
print(a1dd ** 2)                    # square every element
print(np.square(a1dd))                # same result, using a NumPy function
print(a2ddd % 2)                        # modulus -- remainder after dividing by 2


# =========================================================================
# 7. AGGREGATIONS
# =========================================================================
# An aggregation reduces a large array down to a single summary value.

print(a1d)
print(sum(a1d))          # Python's built-in sum() -- works, but SLOWER on big arrays
print(np.sum(a1d))         # NumPy's sum() -- optimized & much FASTER, prefer this

lambI_array = np.random.random(1000)   # 1000 random floats between 0 and 1
print(lambI_array)
print(lambI_array.size)     # 1000
print(np.sum(lambI_array))    # total sum

# %timeit is a Jupyter/IPython "magic command" -- only works inside a notebook,
# not in a plain .py script. It benchmarks how fast a line of code runs.
# %timeit sum(lambI_array)
# %timeit np.sum(lambI_array)

print(np.max(a2d))    # largest value in the array
print(np.min(a2d))     # smallest value in the array


# =========================================================================
# 8. STATISTICS
# =========================================================================
# Mean      = sum of all values / how many values there are
# Variance  = how spread out the values are from the mean
# Std (STD) = square root of the variance (same units as original data)

print(np.mean(a2d))              # Mean
print(np.var(a2d))                 # Variance
print(np.sqrt(np.var(a2d)))          # Std computed manually from variance
print(np.std(a2d))                     # Std computed directly -- same result


# =========================================================================
# 9. MATRIX MULTIPLICATION
# =========================================================================

np.random.seed(seed=0)
matrix1 = np.random.randint(10, size=(4, 4))
print(matrix1)

np.random.seed(seed=0)
matrix2 = np.random.randint(29, size=(4, 4))
print(matrix2)

# a) Element-wise multiplication -- requires matching shapes
print(matrix1 * matrix2)

# b) Dot product -- true matrix multiplication.
# Rule: columns of the first matrix must equal rows of the second matrix.
print(matrix1)

np.random.seed(seed=0)
matrix3 = np.random.randint(29, size=(4, 3))
print(matrix3)

print(matrix1.dot(matrix3))   # (4,4) dot (4,3) -> result is (4,3)

# If shapes don't naturally line up, use .T (transpose) to flip rows/columns
np.random.seed(seed=0)
matrix4 = np.random.randint(29, size=(4, 4))
print(matrix4)

print(matrix1.dot(matrix4.T))   # transpose makes the shapes compatible


# =========================================================================
# 10. RESHAPE AND TRANSPOSE
# =========================================================================

matrix4 = np.random.randint(29, size=(4, 3))
print(matrix4.T)          # transpose: flips rows and columns
print(matrix4)
print(matrix4.shape)        # (4, 3)
print(matrix4.T.shape)        # (3, 4)

# Reshape: changes the array's shape while keeping the same total elements
print(matrix4.reshape(2, 3, 2))
print(matrix4.reshape(2, 3, 2).shape)   # (2, 3, 2)


# =========================================================================
# 11. PRACTICAL EXAMPLE -- NUMPY + PANDAS (weekly sales)
# =========================================================================

np.random.seed(seed=0)
matrix5 = np.random.randint(29, size=(5, 3))    # 5 days x 3 items -- quantities sold
print(matrix5)

price = np.array([27, 53, 54])    # price per item (item1, item2, item3)
print(price)

# Turn the price array into a labeled 1-row DataFrame
price_df = pd.DataFrame(price.reshape(1, 3),
                         index=['Price'],
                         columns=['item1', 'item2', 'item3'])
print(price_df)
print(price_df.shape)   # (1, 3)

print(matrix5.T)

# Dot product: (1x3) prices . (3x5) transposed quantities = (1x5) total cost per day
total = price_df.dot(matrix5.T)
print(total)

# Turn the quantities into a labeled DataFrame, one row per weekday
weekly_sales = pd.DataFrame(matrix5,
                             index=['mon', 'tue', 'wed', 'thurs', 'fri'],
                             columns=['item1', 'item2', 'item3'])
print(weekly_sales)

# Add a "Total ($)" column: total is shape (1,5); we transpose it to (5,1)
# so it lines up row-for-row with weekly_sales, then flatten it into a plain
# 1D array so pandas can assign it as a new column cleanly.
weekly_sales['Total ($)'] = total.T.values.flatten()
print(weekly_sales)


# =========================================================================
# 12. COMPARISON OPERATORS
# =========================================================================
# Comparisons broadcast element-by-element and return True/False arrays.

print(a1d)
print(a2d)
print(a1d > a2d)     # element-wise "greater than"
print(a1d < a2d)       # element-wise "less than"
print(a1d > 3)           # compare every element against a single number
print(a1d <= a2d)         # less-than-or-equal
print(a1d >= a2d)          # greater-than-or-equal
print(a1d == a2d)            # equality check, element-by-element

print(a1d / a2d)
print(a1d // a2d)
print(a1d * a2d)
print(a1d % a2d)


# =========================================================================
# 13. SORTING ARRAYS
# =========================================================================

matrix6 = np.random.randint(29, size=(5, 3))
print(matrix6)

print(np.sort(matrix6))                 # sorts each ROW ascending (default: last axis)
print(np.sort(matrix6, axis=0))           # sorts each COLUMN ascending
print(np.sort(matrix6, axis=1))            # sorts each ROW ascending (explicit)

print(np.argsort(matrix6))                   # returns the INDEXES that would sort the array
print(np.argmin(matrix6, axis=0))              # index of the smallest value in each column
print(np.argmax(matrix6, axis=0))                # index of the largest value in each column

print(matrix6)


# =========================================================================
# 14. READING IMAGES AS ARRAYS
# =========================================================================
# An image is just a NumPy array of pixel numbers (Red, Green, Blue[, Alpha]).
# This script generates a small placeholder image so the code runs
# end-to-end anywhere, without depending on an external image file.

import matplotlib
matplotlib.use('Agg')          # non-interactive backend so this runs outside Jupyter too
import matplotlib.pyplot as plt
from matplotlib.image import imread

# --- Generate a placeholder image (stand-in for a real photo) ---
height, width = 100, 150
placeholder = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        placeholder[i, j] = [int(255 * i / height), int(255 * j / width), 150]
plt.imsave('sample_image.png', placeholder)

img_data = imread('sample_image.png')   # reads the image straight into a NumPy array
print(img_data)

plt.figure()
plt.imshow(img_data)     # displays the array as an image
plt.savefig('sample_image_preview.png')   # saved instead of plt.show() for non-interactive runs

print(img_data.shape)    # (height, width, channels) -- image = a grid of numbers


if __name__ == "__main__":
    print("NumPy fundamentals script loaded and ran successfully.")
