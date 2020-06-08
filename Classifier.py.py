import random
import decimal
from collections import Counter
from itertools import chain
from math import sqrt
from math import pi
from math import exp
import csv
import subprocess
import sys
from math import sqrt


def mean(val):
    a = sum(val)/float(len(val))
    return a

def standard_deviation(val):
    m = mean(val)
    return sqrt(sum([(x-m)**2 for x in val]) / float(len(val)-1))

def class_sep(data):
    new = dict()
    
    for i in range(len(data)):
        x = data[i]
        val = x[0]
        
        if (val in new):
        	pass
        else:
            new[val] = []
       
        new[val].append(x)
    
    return new


def summary_class(data):
    new = dict()
    
    for i in range(len(data)):
        x = data[i]
        val = x[0]
        
        if (val in new):
        	pass
        else:
            new[val] = []
       
        new[val].append(x)
    out_dict = dict()
   
    for value, rows in new.items():
        out_dict[value] = [(mean(column), standard_deviation(column), len(column)) for column in zip(*rows)]  
        del(out_dict[value][0])
    
    return out_dict

def probability(x, mean, standard_deviation):
    if standard_deviation == 0:
        return 0
    else:
        return (1 / (sqrt(2 * pi) * standard_deviation)) * (exp(-((x-mean)*2 / (2 * standard_deviation*2 ))))

def class_prob(summary, row):
    total_rows = sum([summary[label][0][2] for label in summary])
    prob1 = dict()
    
    for value, class_s in summary.items():
        prob1[value] = summary[value][0][2]/float(total_rows)

        for i in range(len(class_s)):
            mean, standard_deviation, _ = class_s[i]
            prob1[value] = prob1[value] * probability(row[i], mean, standard_deviation)
    
    return prob1

def accuracy_(actual, predicted):
    correct = 0
    
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct = correct + 1
   
    return correct / float(len(actual)) * 100.0

def algorithm(train, test):
   
    data = [train, test]
    scores = []
    print('############  \n ')
    print("Starting to Train on 80 data points . . . \nTraining Complete  \n  \n  \n")
    for iter1 in data:
        summarize = summary_class(train)
        pred1 = []
        for r in test:
            output = predict(summarize, r)
            pred1.append(output)

        pred = pred1

        actual = [r[0] for r in iter1]
        acc = accuracy_(actual, pred)
        scores.append(acc)

    return scores


def predict(summ, row):

    prob1 = class_prob(summ, row)
    label, prob = None, 0
    for value, probability in prob1.items():
        if label is None or probability > prob:
            prob = probability
            label = value
    return label
 

with open('Spect_train.txt') as f:
    file1 = f.read().splitlines()

data = []
for list_in in file1:
    list_out = []
    list_in = list_in.replace(',', '')
    for string in list_in:
        list_out.append(int(string))
    
    data.append(list_out)

with open('Spect_test.txt') as f:
    test_file = f.read().splitlines()

test = []
for list_in in test_file:
    list_out = []
    list_in = list_in.replace(',', '')
    for string in list_in:
        list_out.append(int(string))
    
    test.append(list_out)


summary = summary_class(data)

prob1 = class_prob(summary, data[0])

scores = algorithm(data, test)
print('Testing on 187 data point . . . \n')
print(f'Total Accuracy: = {scores[1]}')
print('#############')
