import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



plt.style.use('seaborn')
dp = pd.read_csv("./LogisticRegressionData/mnist_test.csv")
data = dp.values
data = data[:100, :]

# print(data[10])
# print(data.shape)

dt = np.array(data)
np.random.shuffle(dt)
# print(data.shape)

Y = dt[:, 0]
X = dt[:, 1:]
# print(X.shape)
# print(Y.shape)
# print(X[100])
# print(Y[100])

x_train = np.array(X)
y_train = np.array(Y)
# print(x_train.shape)
# print(y_train.shape)





def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def hypo(x, w, b):
    h = np.dot(x, w) + b
    return sigmoid(h)


def error(y_true, x, w, b):
    m = x.shape[0]
    err = 0.0
    for ix in range(m):
        hx = hypo(x[ix], w, b)
        err += y_true[ix] * np.log2(hx) + (1 - y_true[ix]) * np.log2(1 - hx)

    return -err / m


def get_grad(y_true, x, w, b):
    grad_w = np.zeros(x.shape[1])
    grad_b = 0.0

    m = x.shape[0]
    n = x.shape[1]
    for i in range(m):
        hx = hypo(x[i], w, b)
        grad_b += -1 * (y_true[i] - hx)
        for j in range(n):
            grad_w[j] += -1 * (y_true[i] - hx) * x[i][j]

    grad_w /= m
    grad_b /= m
    return [grad_w, grad_b]


def grad_Descent(x, y_true, w, b, lr):
    err = error(y_true, x, w, b)

    [grad_w, grad_b] = get_grad(y_true, x, w, b)
    b = b - lr * grad_b
    for i in range(w.shape[0]):
        w[i] = w[i] - lr * grad_w[i]

    return err, w, b


y_unik = np.unique(y_train)
# print(y_unik)
ws = []
bs = []
ers = []
for index in y_unik:
    y_true = np.array(Y)

    y_true[y_true == index] = 20
    y_true[y_true != 20] = 30

    y_true[y_true == 20] = 1
    y_true[y_true == 30] = 0
    x_true = X[:]

    W = 3 * np.random.random(x_train.shape[1], )
    B = 5 * np.random.random()

    for itr in range(20):
        err, W, B = grad_Descent(x_true, y_true, W, B, lr=0.5)

    ers.append(err)
    ws.append(W)
    bs.append(B)



ws = np.array(ws)
bs = np.array(bs)
# print(ws.shape)
# print(bs.shape)


def draw(x):
    image = x.reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.show()


class_table = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

#Predicting

def predict(x):
    prob = []
    for i in range(bs.shape[0]):
        pro = hypo(x, ws[i], bs[i])
        prob.append(pro)

    prob = np.array(prob)
    ind = np.argmax(prob)
    print(ind)
    print(class_table[ind])
    draw(x)



#Query for Manual Testing
x = X[17]
x = np.array(x)
y = Y[17]
# print(x.shape)
print(y)

predict(x)

