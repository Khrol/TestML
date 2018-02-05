
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('./data/passenger_info.csv')

data.columns


# In[ ]:


ages = data[data.Age.notnull()].Age
ages.mean()


# In[ ]:


features = []

# Sex
array, levels = pd.factorize(data.Sex)
data['factorized_sex'] = array
features.append('factorized_sex')

# Age
data['known_age'] = data.Age.notnull().astype(int)
features.append('known_age')
for age in [10, 20, 30, 40, 50, 60, 70]:
    name = 'more_{}_years'.format(age)
    data[name] = (data['Age'] >= age).astype(int)
    features.append(name)

levels


# In[ ]:


X = data[features]
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

sorted_features = list(sorted(zip(features, reg.coef_), key=lambda x: -abs(x[1])))
sorted_features


# In[ ]:


y_predicted = reg.predict(X_test)


# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
roc_auc = metrics.auc(fpr, tpr)


# In[ ]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# Fare
min_max_scaler = preprocessing.MinMaxScaler()
data['scaled_fare'] = min_max_scaler.fit_transform(data[['Fare']])
features.append('scaled_fare')    
    
# Class
for cl_num in [1, 3]:
    name = 'class{}'.format(cl_num)
    data[name] = (data['Pclass'] == cl_num).astype(int)
    features.append(name)

for sp in [1,2,3,4,5]:
    name = 'sib_sp_{}'.format(sp)
    data[name] = (data.SibSp == sp).astype(int)
    features.append(name)
    
for emb in ['C', 'Q', 'S']:
    name = 'embarked{}'.format(emb)
    data[name] = (data.Embarked == emb).astype(int)
    features.append(name)
    


# In[ ]:


X = data[features]
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

sorted_features = list(sorted(zip(features, reg.coef_), key=lambda x: -abs(x[1])))
sorted_features


# In[ ]:


y_predicted = reg.predict(X_test)


# In[ ]:


fpr_full, tpr_full, _ = metrics.roc_curve(y_test, y_predicted)
roc_auc_full = metrics.auc(fpr_full, tpr_full)


# In[ ]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot(fpr_full, tpr_full, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_full)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


features_part = list(map(lambda x: x[0], sorted_features[:9]))
features_part


# In[ ]:


X = data[features_part]
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

reg.coef_


# In[ ]:


y_predicted = reg.predict(X_test)


# In[ ]:


fpr_part, tpr_part, thresholds = metrics.roc_curve(y_test, y_predicted)
roc_auc_part = metrics.auc(fpr_part, tpr_part)


# In[ ]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot(fpr_full, tpr_full, color='red',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_full)
plt.plot(fpr_part, tpr_part, color='green',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc_part)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


from sklearn.externals import joblib


# In[ ]:


joblib.dump(reg, 'model.pkl')


# In[ ]:


fpr_part, tpr_part, thresholds


# In[ ]:


[thresholds[i] for i in range(len(fpr_part)) if fpr_part[i] > 1 - tpr_part[i]]

