import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

db = np.loadtxt("pcos_dataset.csv", delimiter=',', skiprows=1)
db = np.random.permutation(db)
num_of_data = db.shape[0]
w = np.zeros((db.shape[1],1))

db_x = []
db_y = []

for i in range(num_of_data):
    db_x.append(db[i][:5])
    db_y.append(db[i][5:])

db_x = np.array(db_x)
db_y = np.array(db_y)

db_x = (db_x - np.mean(db_x, axis=0)) / np.std(db_x, axis=0)
print(db_x)
db_x = np.hstack((np.ones((db_x.shape[0], 1)), db_x))

traning_data_x = db_x[0:int(num_of_data*.8)]
traning_data_y = db_y[0:int(num_of_data*.8)]
testing_data_x = db_x[int(num_of_data*.8):]
testing_data_y = db_y[int(num_of_data*.8):]

def sigmoid(z):
    """
    Gives the chance someone has PCOS
    """
    return 1/(1 + np.exp(-z))


def comput_cost(x, y, w):
    z = np.dot(x, w)
    ho = sigmoid(z)
    size = len(y)
    cost = 1/size * np.sum(y* np.log(ho) + (1-y)*np.log(1-ho))
    return cost


def gradient_descent(x, y, w, learning_rate, i):
    m = len(y)
    cost_record = list()

    for _ in range(i):
        z = np.dot(x, w)
        prediction = sigmoid(z)

        graident = 1/m* np.dot(x.T, (prediction - y))

        w = w - learning_rate * graident
        
        cost_record.append(comput_cost(x, y, w))
    return w, cost_record


learning_rate = 0.1
iteration = 10000

w, cost_reocrd = gradient_descent(traning_data_x, traning_data_y, w, learning_rate, iteration)

def prediction(x, w):
    prob = sigmoid(np.dot(x, w))
    return(prob >= 0.5).astype(int)

final = prediction(traning_data_x, w)
test = prediction(testing_data_x, w)

accuracy = accuracy_score(traning_data_y, final)
testac = accuracy_score(testing_data_y, test)

print(accuracy)
print(testac)

np.savetxt("Weights.csv", w, delimiter=',')