import numpy as np

def matrixes():
    f = open("TrainingDat1.txt","r")
    x=[]
    y=[]
    while True:
        line=f.readline().split()
        if len(line) == 0:
            break
        line.insert(0,"1")
        y.append(line.pop(4))
        x.append(line)

    for i in range(len(x)):
        y[i] = float(y[i])
        for j in range(len(x[i])):
            x[i][j] = float(x[i][j])

    xnp = np.array(x)   #creating matrix
    ynp = np.array(y)
    return (xnp,ynp)

matrixes = matrixes()
x = matrixes[0]
y = matrixes[1]
x_t = np.transpose(x) #x transpose
w = (np.linalg.inv(x_t.dot(x)).dot(x_t)).dot(y) #vector w calculation, normal eqn
print("w =", w)
print("prediction result =", np.array([1, 6, 5, 8]).dot(w))
