
import numpy as np
import pandas as pd
from LogisticRegression_class import LogisticRegression

df = pd.read_csv('./titanicdata.csv')

X = df.drop(['Survived'],axis=1)
Y = df['Survived']

X_pre = (X - X.mean(axis=0))/(X.max(axis=0) - X.min(axis=0))
Y_pre = np.where((Y==0),-1,+1)

x0 = np.ones((X_pre.shape[0],1))
X_pre.insert(0,'x0',x0,True)
Y_pre = pd.DataFrame(Y_pre)

X_train, X_valid, X_test = np.split(X_pre.sample(frac=1 , random_state=42)
                                    , [int(0.6*len(X_pre)), int(0.8*len(X_pre))])

Y_train, Y_valid, Y_test = np.split(Y_pre.sample(frac=1, random_state=42)
                                    , [int(0.6*len(Y_pre)), int(0.8*len(Y_pre))])


# array
X_train = np.array(X_train)
X_valid = np.array(X_valid)
X_test = np.array(X_test)

Y_train = np.array(Y_train)
Y_valid = np.array(Y_valid)
Y_test = np.array(Y_test)



# Tune learning rate
min_loss = np.inf
best_lr = 0

for i in range(200):
    model = LogisticRegression(X_train.shape[1])
    lr = 10 ** np.random.uniform(-1,0)
    loss_hist = model.train(X_train,Y_train,lr)
    valid_loss = model.loss(X_valid,Y_valid)
    print('Valid_loss = ',valid_loss,' , learning_rate = ',lr)
    
    if (valid_loss < min_loss):
        min_loss = valid_loss
        best_lr = lr


epochs = 100

# model

X_train1 = np.concatenate((X_train,X_valid),axis=0)
Y_train1 = np.concatenate((Y_train,Y_valid),axis=0)

model = LogisticRegression(X_train1.shape[1])

loss_history = model.train(X_train1, Y_train1, best_lr,epochs)

Y_pred = model.predict(X_test)
loss = model.loss(X_train1,Y_train1)
accuracy = np.sum(Y_test == Y_pred)/Y_test.shape[0] * 100     # nearly 82%
print('Accuracy = ', accuracy)