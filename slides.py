
# coding: utf-8

# <h1>Как тестировать<br/>
# Machine Learning и Artificial Intelligence?</h1>
# 
# <br/>
# <br/>
# 
# ### Игорь Хрол, Минск

# <div>
#   <div style="float: left; width: 60%;">
#       <h1>Кто перед вами?</h1>
#       <ul>
#     <li>Игорь Хрол</li>
# <li>Team Lead / QA Engineer в отделе аналитики Toptal</li>
# <li>\>10 лет в отрасли</li>
# <li>Инженер, тимлид, менеджер, архитектор, тренер, консультант</li>
# <li>Python, Scala, Ruby, Java, SQL и другое</li>
#           <li><a>www.khroliz.com</a></li>
#                 </ul>
#   </div>
#   <div style="float: left; width: 40%;">
#     ![avatar](images/avatar.jpg)
#    </div>
# </div>
# 
# 
# 

# # Где скачать?
# 
# [github.com/Khrol/TestML](https://github.com/Khrol/TestML)
# 
#   <div style="float: left; width: 50%;">
# ![QR code](images/repo_qr.jpg)      
#   </div>
#   <div style="float: left; width: 50%;">
# ![Star repo](images/star_video.gif)
#     </div>

# ![joke](images/joke.jpg)

# In[ ]:


42-41


# <div style="float: left; width: 50%;">
# <h1>План</h1>
# 
# <h2>I. Research</h2>
# 
# <h2>II. Development</h2>
# 
# <h2>III. Production</h2>
# </div>
#   <div style="float: left; width: 50%;">
# ![Plan](images/plan.png)
#     </div>

# # Что такое Machine Learning?
# 
# <div>
# <div style="text-align: center;">
# класс методов искусственного интеллекта, характерной чертой которых является не прямое решение задачи, а обучение в процессе применения решений множества сходных задач
# </div>
# <div style="height: 300px;">
# <img src='images/wiki_logo.png' style="max-height: 100%;"/>
#     </div>
# <div>

# ![titanic](images/titanic.jpg)

# # I. Research
# 
# ![research](images/research.jpg)

# # Что используем?

# In[ ]:


import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

import matplotlib.pyplot as plt


# ## Чтение данных

# In[ ]:


data = pd.read_csv('./data/passengers_info.csv')


# In[ ]:


data.columns


# 
# ## Создание baseline
# 
# ![baseline](images/baseline.jpg)

# ## Создание baseline

# In[ ]:


features_dataframe = pd.DataFrame()


# In[ ]:


array, levels = pd.factorize(data.Sex)
features_dataframe['factorized_sex'] = array


# In[ ]:


features_dataframe['known_age'] = data.Age.notnull().astype(int)
features_dataframe


# ## Обучающие и тестовые выборки
# 
# ![train_test](images/train_test_data.jpg)

# ## Обучающие и тестовые выборки

# In[ ]:


X = features_dataframe
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


(len(X), len(X_train), len(X_test))


# 
#   <div style="float: left; width: 50%;">
#   <h2>Логистическая регрессия</h2>
#   <br/>
# $$y = f(\vec{w}\cdot\vec{x}) = f\left(\sum_j w_j x_j\right)$$  
# </br>
# $$f(t)=\frac{1}{1+e^{-t}} $$
# </div>
#   <div style="float: left; width: 50%;">
# ![perseptron](images/perceptron.png)
# ![logistic_curve](images/logistic_curve.png)
# </div>
# 
# 

# ## Линейная регрессия

# In[ ]:


reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)
reg.coef_


# # Оценка результата

# In[ ]:


y_predicted = reg.predict_proba(X_test).T[1]


# In[ ]:


y_predicted


# # Оценка результата
# 
# 
# 
#   <div style="float: left; width: 60%;">
# <h2>Кривая ошибок</h2>
# <ol>
#     <li>TP (True Positives)</li>
#     <li>TN (True Negatives)</li>
#     <li>FN (False Negatives)</li>
#     <li>FP (False Positives)</li>
# </ol>
#     $$TPR = \frac{TP}{TP + FN}$$
#     <br/>
#     $$FPR = \frac{FP}{TN + FP}$$
# </div>
#   <div style="float: left; width: 40%;">
# ![roc](images/roc_curves.png)
# </div>
# 

# # Оценка результата

# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
roc_auc = metrics.auc(fpr, tpr)

def init_plt():
    plt.figure(figsize=(14,7))
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    
roc_auc


# # Оценка результата

# In[ ]:


init_plt()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.legend(loc="lower right")
plt.show()


# # Feature engineering: scaling

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
features_dataframe['scaled_fare'] = min_max_scaler.fit_transform(data[['Fare']])


# In[ ]:


features_dataframe


# # Feature engineering: категориальные признаки

# In[ ]:


for cl_num in [1, 2]:
    name = 'class{}'.format(cl_num)
    features_dataframe[name] = (data['Pclass'] == cl_num).astype(int)

for sp in [1, 2, 3, 4]:
    name = 'sib_sp_{}'.format(sp)
    features_dataframe[name] = (data.SibSp == sp).astype(int)
    
for emb in ['C', 'Q', 'S']:
    name = 'embarked{}'.format(emb)
    features_dataframe[name] = (data.Embarked == emb).astype(int) 
    
for parch in [0, 1, 2]:
    name = 'parch{}'.format(parch)
    features_dataframe[name] = (data.Parch == parch).astype(int) 
    
for age in [10, 20, 30, 40, 50, 60, 70]:
    name = 'more_{}_years'.format(age)
    features_dataframe[name] = (data['Age'] >= age).astype(int)


# In[ ]:


features_dataframe


# # Повторяем на большем числе признаков

# In[ ]:


X = features_dataframe
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)


# In[ ]:


y_predicted = reg.predict_proba(X_test).T[1]


# In[ ]:


fpr_full, tpr_full, _ = metrics.roc_curve(y_test, y_predicted)
roc_auc_full = metrics.auc(fpr_full, tpr_full)
roc_auc_full


# In[ ]:


init_plt()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot(fpr_full, tpr_full, color='red', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_full)
plt.legend(loc="lower right")
plt.show()


# # Больше не значит лучше

# In[ ]:


sorted_features = list(sorted(zip(features_dataframe.columns, reg.coef_[0]),
                              key=lambda x: -abs(x[1])))
sorted_features


# # Пересматриваем модель на 10-ми признаках

# In[ ]:


features_part = list(map(lambda x: x[0], sorted_features[:10]))
features_part


# In[ ]:


X = features_dataframe[features_part]
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)

reg.coef_


# # Пересматриваем модель на 10-ми признаках

# In[ ]:


y_predicted = reg.predict_proba(X_test).T[1]


# In[ ]:


fpr_part, tpr_part, thresholds = metrics.roc_curve(y_test, y_predicted)
roc_auc_part = metrics.auc(fpr_part, tpr_part)
roc_auc_part


# In[ ]:


init_plt()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot(fpr_full, tpr_full, color='red', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_full)
plt.plot(fpr_part, tpr_part, color='green', lw=2, label='ROC curve (area = %0.4f)' % roc_auc_part)
plt.legend(loc="lower right")
plt.show()


# # Выбор threshold

# In[ ]:


fpr_part, tpr_part, thresholds


# # Выбор threshold

# In[ ]:


for i in range(1, len(fpr_part)):
    if fpr_part[i - 1] < 1 - tpr_part[i - 1] and fpr_part[i] > 1 - tpr_part[i]:
        print(thresholds[i - 1], thresholds[i])
        break


# # II. Development
# 
# ![development](images/development.jpg)

# # Сохраняем модель

# In[ ]:


joblib.dump(reg, 'model.pkl')


# ![demo](images/demo.png)

# # Сравниваем с Research

# In[ ]:


import requests
import json

all_predicted_y = reg.predict_proba(X).T[1]
EPS = 0.000001

def test_record(record):
    id = record['PassengerId']
    actual_result = requests.post('http://localhost:5000/classify',
                    headers={'Content-Type': 'application/json'}, 
                    data=json.dumps({'sex': record['Sex'], 'sib_sp': record['SibSp'], 'age': record['Age'],
                                   'ticket_class': record['Pclass'], 'parch': record['Parch'], 
                                   'embarked': record['Embarked']}))
    actual_score = actual_result.json()['score']
    expected_score = all_predicted_y[id - 1]
    print(record['PassengerId'], expected_score, actual_score)
    assert abs(expected_score - actual_score) < EPS

data.sample(10).apply(test_record, axis=1)


# # III. Production
# 
# ![production](images/production.jpg)

# <div style="float: left; width: 60%;">
# <h1>Воспроизводимость<br/> логов</h1>
# 
# </div>
#   <div style="float: left; width: 40%; height: 600px;">
#   <img src='images/repeat_logs.jpg' style="max-height: 100%;"/>
#     </div>

# # Устаревание модели
# 
# ![ships](images/ships.png)

# # Другие материалы
# 
# - [www.youtube.com/watch?v=T_YWBGApUgs&t=21524s](https://www.youtube.com/watch?v=T_YWBGApUgs&t=21524s)
# - [www.eecs.tufts.edu/~dsculley/papers/ml_test_score.pdf](https://www.eecs.tufts.edu/~dsculley/papers/ml_test_score.pdf)
# - [kaggle.com](http://kaggle.com/)
# - [ods.ai](http://ods.ai/)

# <div style="float: left; width: 70%;">
# <h1>Спасибо за внимание!<br/>Вопросы?<br/></h1>
# <h2>Игорь Хрол</h2>
# <h2>[khroliz@gmail.com](khroliz@gmail.com)</h2>
# <h2>[github.com/Khrol/TestML](https://github.com/Khrol/TestML)</h2>
# 
# </div>
#   <div style="float: left; width: 30%; height: 600px;">
#   <img src='images/question.png' style="max-height: 100%;"/>
#     </div>
