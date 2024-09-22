import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
import random

seed = 1

data = pd.read_csv('train.csv')
data.fillna(0,inplace=True) 
X=data[['Широта', 'Долгота', 'Индекс', 'Площадь',
       'Этаж', 'Размер_участка',
       'Кво_вредных_выбросов',
       'Кво_комнат', 'Кво_спален', 'Кво_ванных', 'Кво_фото', 'Нлч_парковки',
        'Нлч_балкона', 'Нлч_террасы', 'Нлч_подвала',
       'Нлч_гаража', 'Нлч_кондиционера', 'Последний_этаж', 'Верхний_этаж']]
Y = data['Цена']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.01,random_state=1)
knn = KNeighborsRegressor(n_neighbors=16)
knn.fit(x_train, y_train)
to_pred = pd.read_csv('public_test.csv')
to_pred.fillna(0, inplace=True)
x_to_pred_all = to_pred[['id','Широта', 'Долгота', 'Индекс', 'Площадь',
       'Этаж', 'Размер_участка', 'Расход_тепла',
       'Кво_вредных_выбросов',
       'Кво_комнат', 'Кво_спален', 'Кво_ванных', 'Кво_фото', 'Нлч_парковки',
       'Нлч_почтового_ящика', 'Нлч_балкона', 'Нлч_террасы', 'Нлч_подвала',
       'Нлч_гаража', 'Нлч_кондиционера', 'Последний_этаж', 'Верхний_этаж']]
x_to_pred = to_pred[['Широта', 'Долгота', 'Индекс', 'Площадь',
       'Этаж', 'Размер_участка',
       'Кво_вредных_выбросов',
       'Кво_комнат', 'Кво_спален', 'Кво_ванных', 'Кво_фото', 'Нлч_парковки',
        'Нлч_балкона', 'Нлч_террасы', 'Нлч_подвала',
       'Нлч_гаража', 'Нлч_кондиционера', 'Последний_этаж', 'Верхний_этаж']]
y_pred = knn.predict(x_to_pred)
ans = [[int(x_to_pred_all['id'][i]),int(y_pred[i])] for i in range(len(y_pred))]
with open('public_sample_submission_seed_1.csv','w',newline='') as f:
    csv.writer(f).writerow(['id', 'Цена'])
    csv.writer(f).writerows(ans)