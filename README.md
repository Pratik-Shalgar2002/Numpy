# Numpy In Python

## What is Numpy ?

**NumPy** is a Python library used for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

## What is Numpy Array ?
A **NumPy array** is a grid of values, all of the same type, indexed by a tuple of non-negative integers. It is similar to a Python list but is more powerful because it supports efficient operations and broadcasting, allowing for element-wise computations.


## Type of Numpy Array

1. **1D Array (Vector)**: A single-dimensional array, similar to a list.
2. **2D Array (Matrix)**: A two-dimensional array, like a table of rows and columns.
3. **3D Array (Tensor)**: A multi-dimensional array with three or more dimensio
4. **Multi Dimentional**ns.


## Numpy VS Python List
*Advantanges of Numpy*
1. Numpy consumes less memory
2. Numpy is fast
3. Numpy is convenient to use
   

## Installation
```
pip install numpy
```

## Import
```
import numpy as np
```

## Imporatance of numpy in python

1. Wide varity of Mathematical operations
2. it supplies enormous library of high mathematical functions

## Difference between Numpy and Python List


1. **Performance**: NumPy arrays are faster and more efficient than Python lists due to their fixed type and optimized operations.
2. **Memory Usage**: NumPy arrays consume less memory compared to Python lists.
3. **Functional Capabilities**: NumPy supports vectorized operations, allowing for element-wise computations without loops, whereas Python lists require iteration.
4. **Data Type**: All elements in a NumPy array must be of the same type, while Python lists can hold mixed data types.
5. **Built-in Methods**: NumPy has numerous built-in mathematical functions for arrays, which are not available for Python lists.


## Creating Numpy Array
```
var = np.array([1,2,3,4])
print(var)
```

## Dimentions in Array
#1-d Array
```
a = np.array([1,2,3,4])
print(a)
print(a.ndim)
```
#2-D Array
```
b = np.array([[1,2,3,4],[1,2,3,4]])
print(b)
print(b.ndim)
```
#3-D Array
```
c = np.array([[[1,2,3,4],[1,2,3,4],[1,2,3,4]]])
print(c)
print(c.ndim)
```
```
#To create Multi dimentional array use np.array(data,ndmin= value)
d = np.array([1,2,3,4],ndmin=10)
print(d)
print(d.ndim)
```
## Create Numpy array using numpy function
**Zeros**
```
# 1-D
x = np.zeros(5)
print(x)
```
```
#2-D
x = np.zeros((2,3))
print(x)
```
**Ones**
```
#1-D
x = np.ones(5)
print(x)
```
```
#2-D 
x= np.ones((2,3))
print(x)
x= np.ones((3,2))
print(x)
```
**Empty**
```
x= np.empty(4)
print(x)
```
**Dignoal element filled with 1**
```
x = np.eye(3)
print(x)
x = np.eye(3,5)
print(x)
```
**line Space**
```
x = np.linspace(0,20,5)
print(x)
```

## Creating numpy array with random number
1.rand()
```
x = np.random.rand(5)
print(x)
```

2.randn()
```
x = np.random.randn(5)
print(x)
```
3.ranf()
```   
x = np.random.ranf(5)
print(x)
```
4. randint()
```
x = np.random.randint(1, 20, 5)
print(x)
```

## Data Types in NumPy Arrays

NumPy arrays can store elements of a specific data type. The type of data in an array is called `dtype`. Some common NumPy data types include:

1. **int**: Integer types (e.g., `np.int32`, `np.int64`)
2. **float**: Floating point numbers (e.g., `np.float32`, `np.float64`)
3. **complex**: Complex numbers (e.g., `np.complex64`, `np.complex128`)
4. **bool**: Boolean values (`True` or `False`)
5. **str**: String data type (e.g., `np.str_`)

You can specify the data type when creating a NumPy array by using the `dtype` parameter:

```python
arr = np.array([1, 2, 3], dtype=np.float64)
```

**To find Data Type** 
```
x = np.array([1,2,3,4])
print(x.dtype)
```
## Arithmetic Operations
```
a = np.array([1,2,3,4])

b = np.array([1,2,3,4])
```

#### Addition
```

print(a+b)
print(np.add(a,b))
```
#### Subtraction
```
print(a-b)
print(np.subtract(a,b))
```
#### Multiply
```
print(a*b)
print(np.multiply(a,b))
```
#### Divide
```
print(a/b)
print(np.divide(a,b))
```
#### Mod
```
print(a%b)
print(np.mod(a,b))
```
#### power
```
print(a**b)
print(np.power(a,b))
```
#### Reciprocal
```
print(1/a)
print(np.reciprocal(a))
```

## Arithmetic Function
```
x= np.array([2,5,4,7,8,9,2,5,6,10,12,13])
```

**Max**
```

print(np.max(x))
```


**min**

```
print(np.min(x))
```
**Argmin/argmax**
```
print(np.argmin(x))
print(np.argmax(x))
```
**Squre Root**

```
print(np.sqrt(x))
```
**Sin**
```
```
print(np.sin(x))
**cos**
```

print(np.cos(x))
```
**Cumsum**
```
print(np.cumsum(x))
```
## Shape & Reshape
#### Shape
```
import numpy as np
x = np.array([[1,2],[3,4]])
x
print(x.shape)
y = np.array([1,2, 3,4],ndmin = 4)
y
print(y.shape)
```
#### Reshape
```
x = np.array([1,2,3,4,5,6,7,8])
y = x.reshape(4,2)
y
```
## Broad casting
- Should have same dimensions
- Both should have 1 from left hand side
## Indexing
#### 1-D
import numpy as np
```
x = np.array([1,2,3,4])
#             0 1 2 3
print(x[2])
```
#### 2-D
```
import numpy as np
x = np.array([[1,2],[3,4]])       
x
#     0  1 
# 0 [[1, 2],
# 1  [3, 4]]

print(x[0,0])
print(x[0,1])
```
#### 3-D
```
import numpy as np
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       
print(x)
#     0 1
# [ [[1 2]0
#    [3 4]1]0
#   [[5 6]0
#    [7 8]1]1 ]  
print(x[1,0,1])
```

## Slicing
```
import numpy as np
x = np.array([1,2,3,4,5])
print(x)

print(x[0:])
print(x[0:3])
print(x[3:])
```
## Iterating Numpy array
#### method 1
##### 1-D
```
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9])
print(x)
for i in x:
    print(i)
```
##### 2-D
```
import numpy as np

x = np.array([[1,2,3,4],[5,6,7,8]])
print(x)
for i in x:
    print(x)
#False Result
for i in x:
    for j in i:
        print(j)
```
##### 3-D
```
import numpy as np
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       
print(x)
for i in x:
    for j in i:
        for k in j:
            print(k)
```
#### Method 2
```
import numpy as np
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       
print(x)

for i in np.nditer(x):
    print(i)
```
```
# for Index
for i,d  in np.ndenumerate(x):
    print(i,d)
```
## Copy VS View 
#### Copy
- **Definition**: A copy is an entirely new array or DataFrame that replicates the data of the original. Any modifications to the copied object do not affect the original.
```
import numpy as np
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       
print(x)
var = x.copy()
print(var)
a = np.array([1,2,3,4])
print(a)
print()
b = a.copy()
print(b)
print()

a[2] = 30
print(a)
print()
print(b)

```

#### View
- **Definition**: A view is a new object that shares the same data as the original array or DataFrame. Any modifications made to the view will affect the original and vice versa.
```
import numpy as np
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])       
print(x)
var2 = x.view()
print(var2)
a = np.array([1,2,3,4])
print(a)
print()
b = a.view()
print(b)
print()

a[2] = 30
print(a)
print()
print(b)
```
## Join & Split Funtion
- **Join Array**: Joining means putting content of two or more array in single one.
### Concatenate
#### 1-D
```
a = np.array([1,2,3,4])
print(a)
b = np.array([5,6,7,8])
print(b)
c = np.concatenate((a,b))
print(c)
```
#### 2-D
```
a = np.array([[1,2],[3,4]])
print(a)
print()
b = np.array([[5,6],[7,8]])
print(b)
print()
c = np.concatenate((a,b), axis = 1)
print(c)

a = np.array([[1,2],[3,4]])
print(a)
print()
b = np.array([[5,6],[7,8]])
print(b)
print()
c = np.concatenate((a,b), axis = 0)
print(c)
```
### Stack
```
a = np.array([1,2,3,4])
print(a)
print()
b = np.array([5,6,7,8])
print(b)
print()
c = np.stack((a,b),axis = 0)
print(c)
print()
d = np.stack((a,b),axis = 1)
print(d)
print()
e = np.hstack((a,b))
print(e)
print()
f = np.vstack((a,b))
print(f)
print()
g = np.dstack((a,b))
print(g)
print()
```
### Split
- Spliting breaks one array into multiple
```
a = np.array([1,2,3,4,5,6,7,8])
print(a)

b = np.array_split(a,3)
print(b)
c = np.array_split(a,2)
print(c)
```
## Numpy Array Functions
```
x= np.array([1,2,3,4,2,14,45,56,54,78,83,12, 3, 4, 2])
#            0 1 2 3 4 5  6  7   8  9 10 11 12 13 14
```
### Search
```
a = np.where(x == 2)
print(a)
```
### Search Sorted Array
```
g = np.array([1,2,3,4,6,7,8,9,10])
a = np.searchsorted(g, 5)
print(a)

B = np.searchsorted(g, 5, side= "right")
print(B)
```
### Sort Array
```
x = np.array([3,5,2,34,75,3,7,89,6,5])
print(x)
print()
y = np.sort(x)
print(y)
```
### Filter Array
```
e = np.array([1,2,3,4])
f= [True,False,False,True]
print(e[f])
```
### Shuffle
```
x = np.array([1,2,3,4,5,6,7,8,9])

np.random.shuffle(x)
print(x)
```
### Unique
```
x= np.array([1,2,3,4,2,14,45,56,54,78,83,12, 3, 4, 2])
y = np.unique(x)
print(y)
z = np.unique(x, return_index = True)
print(z)
f = np.unique(x, return_index = True, return_counts = True)
print(f)
```
### Resize

```
x = np.array([1,2,3,4,5,6,7,8])

y= np.resize(x,(2,3))
print(y)
z= np.resize(x,(3,2))
print(z)
```
### Flatten & Ravel
```
x = np.array([[1,2,3,4],[5,6,7,8]])
print(x)
z= np.resize(x,(3,2))
print(z)
print("Flatten : ", z.flatten())
print("Ravel : ", np.ravel(z))
```
### Insert
#### 1-D
```
n = np.array([1,2,3,5,6,7,8])
print(n)
# Insert Function
a = np.insert(n,3,4)
print(a)
# Append Funtion
b = np.append(n,4)
print(b)
````
#### 2-D
```
n = np.array([[1,2,3,4],[5,6,7,8]])
print(n)
# Insert Function
a = np.insert(n,1,10,0)
print(a)
b = np.insert(n,1,10,1)
print(b)
# Append Function
c = np.append(n,[[10,10,1,2]],axis = 0)
print(c)
f = np.append(n,[[1],[2]],axis = 1)
print(f)
```
### Delete
```
x = np.array([1,2,3,4,5,5,6,7,8])
print(x)
print(len(x))
y = np.delete(x,4)
print(y)
print(len(y))
```

