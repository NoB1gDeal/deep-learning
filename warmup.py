import numpy as np

a=np.array([4,5,6])
print(f"The type of a:\t{type(a)}")
print(f"The shape of a:\t{a.shape}")
print(f"The first element in a is:\t{a[0]}")

b=np.array([[4,5,6],[1,2,3]])
print(f"The type of b:\t{b.shape}")
print(f"b(0,0):\t{b[0][0]}\nb(0,1):\t{b[0][1]}\nb(0,0):\t{b[1][1]}")

a = np.zeros((3,3),dtype=int)
b = np.ones((4, 5))
c = np.eye(4)
d = np.random.rand(3, 2)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"a:{a}")
print(f"a(2, 3):\t{a[2][3]}\na(0, 0):\t{a[0][0]}")

b = a[0:2, 2:4]
print(f"b:{b}")
print(f"b(0, 0):{b[0][0]}")

c = a[1:3, :]
print(f"c:{c}")
print(f"The last element in the 1st row of c:{c[0][-1]}")

a = np.array([[1, 2], [3, 4], [5, 6]])
print(f"[a(0, 0), a(1, 1), a(2, 0)]: {a[[0, 1, 2], [0, 1, 0]]}")

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(f"[(0,0),(1,2),(2,0),(3,1)]:{a[np.arange(4), b]}")

a[np.arange(4), b] += 10
print(f"a: {a}")

x = np.array([1, 2])
print(f"The type of x: {type(x)}")

x = np.array([1.0, 2.0])
print(f"The type of x: {type(x)}")

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(f"x + y:{x + y}")
print(f"np.add(x, y):{np.add(x, y)}")

print(f"x - y: {x - y}")
print(f"np.subtract():{np.subtract(x, y)}")

print(f"x * y: {x * y}")
print(f"np.multiply(x, y): {np.multiply(x, y)}")
print(f"np.dot(x,y): {np.dot(x,y)}")

# not the square matrix
print('*' * 50)
temp_x = np.array([1, 2, 3])
temp_y = np.array([3, 2, 1])
print(f"x * y: {temp_x * temp_y}")
print(f"np.multiply(x, y): {np.multiply(temp_x, temp_y)}")
print(f"np.dot(x,y): {np.dot(temp_x,temp_y)}")

print(f"x / y: {np.divide(x, y)}")

print(f"np.sqrt(): {np.sqrt(x)}")

print(x.dot(y))
print(np.dot(x,y))

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

print(np.mean(x))
print(np.mean(x,axis = 0))
print(np.mean(x,axis =1))

print(f"TH transpose of x:{x.T}")

print(f"e^x: {np.exp(x)}")

print(np.argmax(x))
print(np.argmax(x, axis=0))
print(np.argmax(x, axis=1))

import matplotlib.pyplot as plt
x = np.arange(0, 100, 0.1)
plt.plot(x, [i * i for i in x])
plt.legend(['x^2'])
plt.show()

import matplotlib.pyplot as plt
x = np.arange(0, 3 * np.pi, 0.1)
plt.plot(x, [np.sin(i) for i in x])
plt.plot(x, [np.cos(i) for i in x])
plt.legend(["sin(x)", "cos(x)"])
plt.show()