# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 09:34:00 2021

@author: XaineChaeson
"""

import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
# from collections import Counter
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('car_loan_train.csv')
test = pd.read_csv('test.csv')
sample_submit = test[['customer_id']]
test['loan_default']=-1

data=pd.concat([train,test])


#删除单一值
for i in data.columns:
    flag=len(data[i].value_counts())
    if flag==1:
        #说明是无效值
        data=data.drop([i],axis=1)
#消除inf
data=data.replace([np.inf, -np.inf], -1)

all_cols = [f for f in data.columns if f not in ['customer_id','loan_default']]
train=data[data['loan_default'] !=-1]
label=train['loan_default']
train=train[all_cols]
test=data[data['loan_default']==-1]
test=test[all_cols]

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def cv_model(clf_name,clf, train_x, train_y, test_x):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_leaves': 2 ** 7,
            'metric': 'auc',
            'min_child_weight': 5,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'seed': seed,
            'n_jobs':-1
        }

        model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], feval=lgb_f1_score,verbose_eval=500,early_stopping_rounds=200)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

lgb_train, lgb_test = cv_model('lgb',lgb, train, label, test)
predicted = data[data['loan_default'] !=-1].loc[:,['customer_id','loan_default']]

# lgb_train.to_csv('predicted/lgb.csv')
predicted['pred'] = lgb_train

# predicted['pred'] = pd.read_csv('predicted/lgb.csv')
predicted['pred'] = predicted['pred'].apply(lambda x:1 if x>0.24 else 0).values

test['loan_default'] = lgb_test
test['loan_default'] = test['loan_default'].apply(lambda x:1 if x>0.24 else 0).values
test_ = pd.read_csv('test.csv')


import matplotlib.pyplot as plt

x = train.index
y = train.loan_to_asset_ratio
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot()

ax.scatter(x[predicted['pred'] != 1], y[predicted['pred'] != 1], s=0.1,alpha=0.7)
ax.scatter(x[predicted['pred'] == 1], y[predicted['pred'] == 1],c='yellow', s=0.3,alpha=0.7)

# ax.scatter(x[predicted['loan_default'] != 1], y[predicted['loan_default'] != 1], s=0.1,alpha=0.7)
# ax.scatter(x[predicted['loan_default'] == 1], y[predicted['loan_default'] == 1],c='red', s=0.3,alpha=0.7)


plt.show()


