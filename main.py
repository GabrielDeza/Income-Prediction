import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import scipy.signal as sig

from model import MultiLayerPerceptron
from dataset import AdultDataset

from util import *
import time


seed = 4

# =================================== LOAD DATASET =========================================== #


data = pd.read_csv('adult.csv')




# =================================== DATA CLEANING =========================================== #


col_names = data.columns
#print(col_names)
for feature in col_names:
    data = data.loc[ data[feature] != "?" ]

# =================================== BALANCE DATASET =========================================== #

balance_num = min(data["income"].value_counts())
data1 = data.loc[ data["income"] != "<=50K" ]
data2 = data.loc[ data["income"] != ">50K" ]
data3 = data2.sample(n = balance_num, random_state = seed)
data = pd.concat((data1,data3))


    ######

# =================================== DATA STATISTICS =========================================== #

categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']


# =================================== DATA PREPROCESSING =========================================== #

continous_feats = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
data_cont = data[continous_feats]

for feature in continous_feats:
    avg = data_cont[feature].mean()
    deviation = data_cont[feature].std()
    data_cont[feature] = (data_cont[feature] - avg) / deviation
data_cont = data_cont.to_numpy()

#initializing categorical labels
cat_feat = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country','income']
#categorical part of data
data_categ = data[cat_feat]
#initiliaze label encoder
label_encoder = LabelEncoder()
#Label encoder everything including income
for feature in cat_feat:
    data_categ[feature] = label_encoder.fit_transform(data_categ[feature])
#saving encoded income then dropping it
data_income = data_categ["income"]
data_categ = data_categ.drop("income", axis = 1)
#passing in data into One Hot Encoding
oneh_encoder = OneHotEncoder(categories="auto")
data_categ = oneh_encoder.fit_transform(data_categ)
#transforming into numpy array
data_categ = data_categ.toarray()
data_income = data_income.to_numpy()
#collating continous and categorical data
data = np.concatenate((data_categ,data_cont),axis =1)
#print(data.shape)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #


train_data, valid_data, train_label, valid_label = train_test_split(data, data_income, test_size = 0.2, random_state =9)
# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):

    train_dataset = AdultDataset(train_data, train_label)
    valid_dataset = AdultDataset(valid_data, valid_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_model(lr):
    x = len(train_data[0])
    model = MultiLayerPerceptron(x)
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer

def evaluate(model, val_loader):
    total_corr = 0
    for i, (batch, labels) in enumerate(val_loader):
        prediction = model(batch)
        corr = (prediction > 0.5).squeeze().long() == labels
        total_corr += int(corr.sum())
    return float(total_corr)/len(val_loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],
                        default='linear')
    args = parser.parse_args()
    #Command Line argument passing throught
    corr_values = []
    time_val = []
    valid_values  = []
    #Initializing Model, Loss Function and Optimizer for given Learning Rate
    model, loss_fnc, optimizer = load_model(args.lr)
    #Loading Training and Validation data (data and labels)
    training, validation = load_data(args.batch_size)
    #Initializing batch number
    t= 0
    #Initiliziation of Plotting Arrays
    #Epoch loop
    start = time.time()
    for epoch in range(args.epochs):
        #initialization of parameters for loss and accuracy
        accum_loss = 0
        tot_corr = 0
        for i, (batch, labels) in enumerate(training): #batch = input data , labels = labels
            #print(f"{epoch}\t{i}")
            optimizer.zero_grad()
            predictions = model(batch)
            batch_loss = loss_fnc(input = predictions.squeeze(),target = labels.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            corr = (predictions > 0.5).squeeze().long() == labels
            tot_corr += int(corr.sum())
            if (t + 1) % args.eval_every == 0:
                valid_acc = evaluate(model, validation)
                corr_values.append(float(int(corr.sum())) / args.batch_size)
                valid_values.append(valid_acc)
                print("Epoch: {}, Step {} | Train acc: {}| Valid acc: {}".format(epoch + 1, t+1, float(int(corr.sum())) / args.batch_size,valid_acc))
                accum_loss = 0
            t = t + 1
    end = time.time()
    diff = end-start
    times = np.linspace(0, diff, num=len(corr_values))
    train_acc = np.array(corr_values)
    valid_acc = np.array(valid_values)
    time_acc = np.array(time_val)
    x = np.arange(1, len(corr_values)+1, 1)
    train_acc = sig.savgol_filter(train_acc,3,1)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for a sigmoid function' )
    plt.ylim((0, 1.1))
    plt.plot(x, train_acc)
    plt.plot(x, valid_acc)
    plt.legend(['Training Data', 'Validation Data'])
    plt.show()
    print("Train acc:{}".format(float(tot_corr) / len(training.dataset)))
    print("highest validation accuracy", max(valid_acc))
    print("time", diff)

if __name__ == "__main__":
    main()
