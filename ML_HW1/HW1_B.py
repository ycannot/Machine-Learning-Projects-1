import numpy as np
f = open("TrainingDat2.txt", "r")
alpha=0.0002
x=[]
y=[]
company={}  #to show corresponding value to brands
modal={}    #to show corresponding value to modal
test_val=np.array([1, 1, 1, 9000])  #data for prediction

def h(x, w):
    return(1/(1+np.exp(-(x.dot(w)))))

def matrixes():     #prepares x and y matrices
    while True:
        line = f.readline()
        if len(line) == 0:
            f.close()
            break
        line=line.strip().split(",")
        y.append(line.pop(0))
        line.insert(0,"1")
        x.append(line)

    for i in range(len(x)):
        y[i] = float(y[i])
        for j in range(len(x[i])):
            if j==1:
                if x[i][j] not in company:
                    company[x[i][j]] = len(company) + 1
                x[i][j] = company[x[i][j]]
            elif j==2:
                if x[i][j] not in modal:
                    modal[x[i][j]] = len(modal) + 1
                x[i][j] = modal[x[i][j]]
            else:
                x[i][j] = float(x[i][j])

    xnp = np.array(x)   #creating matrix
    ynp = np.array(y)
    return (xnp, ynp)

matrixes = matrixes()
x = matrixes[0]
y = matrixes[1]

print("companies= ", company)
print("modal= ", modal)

y_vals = [[], [], [], [], [], [], []] #indexes y_vals = [y(0), y(1), y(2), y(3), y(-1), y(-2), y(-3)]. each element is for binary comparison.
                                      #one vs all comparison.
indexes = [0, 1, 2, 3, -1, -2, -3]
for i in y:
    if i == 0:
        y_vals[0].append(1)
        y_vals[1].append(0)
        y_vals[2].append(0)
        y_vals[3].append(0)
        y_vals[4].append(0)
        y_vals[5].append(0)
        y_vals[6].append(0)
    elif i == 1:
        y_vals[0].append(0)
        y_vals[1].append(1)
        y_vals[2].append(0)
        y_vals[3].append(0)
        y_vals[4].append(0)
        y_vals[5].append(0)
        y_vals[6].append(0)
    elif i == 2:
        y_vals[0].append(0)
        y_vals[1].append(0)
        y_vals[2].append(1)
        y_vals[3].append(0)
        y_vals[4].append(0)
        y_vals[5].append(0)
        y_vals[6].append(0)
    elif i == 3:
        y_vals[0].append(0)
        y_vals[1].append(0)
        y_vals[2].append(0)
        y_vals[3].append(1)
        y_vals[4].append(0)
        y_vals[5].append(0)
        y_vals[6].append(0)
    elif i == -1:
        y_vals[0].append(0)
        y_vals[1].append(0)
        y_vals[2].append(0)
        y_vals[3].append(0)
        y_vals[4].append(1)
        y_vals[5].append(0)
        y_vals[6].append(0)
    elif i == -2:
        y_vals[0].append(0)
        y_vals[1].append(0)
        y_vals[2].append(0)
        y_vals[3].append(0)
        y_vals[4].append(0)
        y_vals[5].append(1)
        y_vals[6].append(0)
    elif i == -3:
        y_vals[0].append(0)
        y_vals[1].append(0)
        y_vals[2].append(0)
        y_vals[3].append(0)
        y_vals[4].append(0)
        y_vals[5].append(0)
        y_vals[6].append(1)


w_vals = [[], [], [], [], [], [], []] #indexes w_vals = [w(0), w(1), w(2), w(3), w(-1), w(-2), w(-3)]. each element is for binary comparison.
                                      #one vs all comparison.
for i in range(len(w_vals)):          #this loop calculates w values for each y0, y1,y2 etc. and saves to w_vals
    temp = np.zeros(len(x[0]))

    for j in range(len(temp)):
        for k in range(len(x)):
            temp[j] -= (alpha/len(x))*(h(x[k],temp)-y_vals[i][k])*x[k][j] #gradient descent

    w_vals[i] = temp
    print("w"+str(i)+"= ", temp)

for i in range(len(w_vals)):
    print("Output of Hw function for",indexes[i], ":", h(test_val, w_vals[i]))
    if i==4:
        print(test_val)
        print(w_vals[i])
        print(test_val.dot(w_vals[i]))

