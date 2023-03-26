#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from collections import Counter
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)

import re
from nltk.stem import WordNetLemmatizer

# Grid search for optimal parameters of the model
from sklearn.model_selection import GridSearchCV

# Model modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# modules for # estimate
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# modules for encoding features
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Modules for dividing a data set
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from gensim.models import word2vec




train = pd.read_json('../input/train.json', encoding = 'UTF-8')
test = pd.read_json('../input/test.json', encoding = 'UTF-8')




df = train.copy()

print(df.shape)
df.head(2)




df.count()




df.isnull().sum()




print('Cuisine is {}.'.format(len(df.cuisine.value_counts())))
df.cuisine.value_counts()




# cuisine type visualization

plt.style.use('ggplot')
df.cuisine.value_counts().plot(kind = 'bar',
                              title='Cuisine Types',
                              figsize=(20,5),
                              legend=True,
                              fontsize=12)
plt.ylabel("Number of Recipes", fontsize=12)
plt.show()




### Work to count ingredients per recipe per row

# Here, the ingredient will be counted one by one.
bag_of_ingredients = [Counter (ingredient) for ingredient in df.ingredients]

# Number of each ingredient by type
sum_of_ingredients = sum (bag_of_ingredients, Counter ())

########################################################################################

### Work to put sum_of_ingredients in dataframe

# dict -> list -> dataframe
sum_of_ingredients_dict = dict (sum_of_ingredients)
sum_of_ingredients_list = list (sum_of_ingredients_dict.items ())

ingredients_df = pd.DataFrame (sum_of_ingredients_list)
ingredients_df.columns = ['ingredient', 'count']
ingredients_df.tail (2)

print ('Before the preprocessing, the total ingredients are {}.'. format (len (ingredients_df)))




ingredients_df.head(20)




def pre_processing_(recipe):
    
    wnl = WordNetLemmatizer()
    
    recipe = [str.lower(ingredient) for ingredient in recipe]
    recipe = [delete_brand_(ingredient) for ingredient in recipe]
    recipe = [delete_state_(ingredient) for ingredient in recipe]
    recipe = [delete_comma_(ingredient) for ingredient in recipe]
    recipe = [original_(ingredient) for ingredient in recipe]
    recipe = [delete_space_(ingredient) for ingredient in recipe]

    return recipe

def delete_brand_(ingredient):
    ingredient = re.sub("country crock|i can't believe it's not butter!|bertolli|oreo|hellmann's"
                        , '', ingredient)
    ingredient = re.sub("red gold|hidden valley|original ranch|frank's|redhot|lipton", '', ingredient)
    ingredient = re.sub("recipe secrets|eggland's best|hidden valley|best foods|knorr|land o lakes"
                        , '', ingredient)
    ingredient = re.sub("sargento|johnsonville|breyers|diamond crystal|taco bell|bacardi", '', ingredient)
    ingredient = re.sub("mccormick|crystal farms|yoplait|mazola|new york style panetini", '', ingredient)
    ingredient = re.sub("ragu|soy vay|tabasco|truvia|crescent recipe creations|spice islands", '', ingredient)
    ingredient = re.sub("wish-bone|honeysuckle white|pasta sides|fiesta sides", '', ingredient)
    ingredient = re.sub("veri veri teriyaki|artisan blends|home originals|greek yogurt|original ranch"
                        , '', ingredient)
    ingredient = re.sub("jonshonville", '', ingredient)

    ingredient = re.sub("oscar mayer deli fresh smoked", '', ingredient)

    return ingredient

def delete_state_(ingredient):

    ingredient = re.sub('frozen|chopped|ground|fresh|powdered', '', ingredient)
    ingredient = re.sub('sharp|crushed|grilled|roasted|sliced', '', ingredient)
    ingredient = re.sub('cooked|shredded|cracked|minced|finely', '', ingredient)        
     return ingredient

def delete_comma_(ingredient):

    ingredient = ingredient.split(',')
    ingredient = ingredient[0]

    return ingredient

def original_(ingredient):

    ingredient = re.sub('[0-9]', '', ingredient)

    ingredient = ingredient.replace("oz.", '')
    ingredient = re.sub('[&%()®™/]', '', ingredient)
    ingredient = re.sub('[-.]', '', ingredient)

    ingredient = wnl.lemmatize(ingredient)

    return ingredient

def delete_space_(ingredient):
    ingredient = ingredient.strip()
    return ingredient




df['ingredients'] = df['ingredients'].apply(lambda x : pre_processing_(x))


df['ingredients_train'] = df['ingredients'].apply(','.join)


tfv = TfidfVectorizer()
X = tfv.fit_transform(df['ingredients_train'].values)

print(list(tfv.vocabulary_.keys())[:10])

print(X.shape)

print(type(X))

print(X[2999])




Lec = LabelEncoder()
train_target_value = Lec.fit_transform(df['cuisine'].values)

print(train_target_value.shape)

print(train_target_value[:20])

print(Lec.classes_)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, train_target_value)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

""" Random Forest Model """
def RandomForestClassifier_():
    
    pipe = Pipeline([('classifier', RandomForestClassifier())])
    hyperparameter_space = [{'classifier' : [RandomForestClassifier()], 
                             'classifier__n_estimators' : [350, 375, 400],
                             'classifier__max_features' : ['sqrt', 'log2']}]
    
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)

    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))
    
""" SVM Model """
def SVM_():
    pipe = Pipeline([('classifier', SVC())])
    hyperparameter_space = [{'classifier': [SVC()],
                             'classifier__gamma': ['auto'],
                             'classifier__C' : [10, 15]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))


""" KNN Model """
def KNN_():
    
    knn = KNeighborsClassifier()
    
    pipe = Pipeline([('classifier', knn)])
    hyperparameter_space = [{'classifier': [knn],
                             'classifier__n_neighbors': [15, 20, 25],
                             'classifier__leaf_size' : [20, 25, 30]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
        
        
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, knn.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))

""" Xgboost Model"""
def Xgboost_():
    pipe = Pipeline([('classifier', xgb.XGBClassifier())])
    hyperparameter_space = [{'classifier': [xgb.XGBClassifier()],
                             'classifier__max_depth': [3, 4, 5],
                             'classifier__n_estimators' : [350, 375, 400]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))


""" Decision Tree Model """
def DecisionTree_():
    pipe = Pipeline([('classifier', DecisionTreeClassifier())])
    hyperparameter_space = [{'classifier': [DecisionTreeClassifier()],
                             'classifier__max_depth': [50, 60, 70]}]
    grid = GridSearchCV(pipe, hyperparameter_space, cv=3)
    grid.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, grid.predict(X_test), digits=4, target_names=cuisine))
    
    return print("Best parameters:\n{}\n".format(grid.best_params_), 
                 "Best score : {}\n".format(grid.best_score_),
                 "Test score : {}".format(grid.score(X_test, y_test)))

def Neural_network_():
    nn = MLPClassifier(hidden_layer_sizes=(400,500,400))
    nn.fit(X_train, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, nn.predict(X_test), digits=4, target_names=cuisine))
    return print("Test score : {}".format(nn.score(X_test, y_test)))




get_ipython().run_cell_magic('time', '', 'rf = RandomForestClassifier_()')




get_ipython().run_cell_magic('time', '', 'SVM = SVM_()')




get_ipython().run_cell_magic('time', '', 'KNN = KNN_()')




get_ipython().run_cell_magic('time', '', 'Xgboost_()')




get_ipython().run_cell_magic('time', '', 'DecisionTree_()')




get_ipython().run_cell_magic('time', '', 'Neural_network = Neural_network_()')




# Feature importance
pd.Series(xgbr.feature_importances_).plot(kind='bar')

all_ingredients = set()
df['ingredients'].apply(lambda x : [all_ingredients.add(i) for i in list(x)])
#print(all_ingredients)

for ingredient in all_ingredients:
    df[ingredient] = df['ingredients'].apply(lambda x : ingredient in x)
    
len(df.columns)




# Copy
df_dummy = df.copy()

del df_dummy['id']
del df_dummy['ingredients']

df_features = df_dummy.copy()

del df_features['cuisine']
df_features.tail(1)

Lec = LabelEncoder()
train_target_value = Lec.fit_transform(df_dummy['cuisine'].values)

print(train_target_value.shape)

print(train_target_value[:10])

print(Lec.classes_)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_features, train_target_value)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

pca.components_

def Xgboost_():
    xgbr = xgb.XGBClassifier(
        n_estimators = 400,
        max_depth = 5
    ).fit(X_train_pca, y_train)
        
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, xgbr.predict(X_test_pca), digits=4, target_names=cuisine))
    return print("Test score : {}".format(xgbr.score(X_test_pca, y_test)))




get_ipython().run_cell_magic('time', '', 'Xgboost = Xgboost_()')




def Neural_network_():
    nn = MLPClassifier(hidden_layer_sizes=(400,500,400))
    nn.fit(X_train_pca, y_train)
    
    cuisine = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek',
               'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan',
               'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']

    print (classification_report(y_test, nn.predict(X_test_pca), digits=4, target_names=cuisine))
    return print("Test score : {}".format(nn.score(X_test_pca, y_test)))




get_ipython().run_cell_magic('time', '', 'Neural_Network = Neural_network_()')


ingredient_list = []

for elements in df['ingredients']:
    ingredient_list.append(elements)
#ingredient_list

ingredient_df = pd.DataFrame(ingredient_list)
#ingredient_df

cuisine_list = []

for element in df['cuisine']:
    cuisine_list.append(element)

ingredient_df.insert(0, "cuisines", cuisine_list)
#ingredient_df.tail(2)

temp = []

for row in ingredient_df.iterrows():
    index, data = row
    temp.append(data.tolist())
    
#temp
new_temp = []

for list_element in temp:
    new_element = [x for x in list_element if x is not None]
    new_temp.append(new_element)
    
#new_temp[0]
model = word2vec.Word2Vec(new_temp, workers = 4, 
                         size = 300, min_count = 3, window = 10)

model.init_sims(replace=True)

model.most_similar('korean')

cuisine_dict = dict(df.cuisine.value_counts().items())
cuisine_list = list(cuisine_dict.keys())

print("Bot: " 
      + "\t" + "I can tell you common ingredients often used in the country." 
      + "\n" + "\t" + "What kind of cuisine do you want to know?" 
      + "\n" + "\t" + "Cuisine list is here."
      + "\n" + "\t" + "====================[CUISINE LISE]===================="
      + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
      + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
      + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
      + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
      + "\n" + "\t" + "======================================================")
    
while True:

    user_question = input("User: " + "\t").strip()
    user_question_lower = user_question.lower()
    
    ingredient_list = []
    for cuisine in cuisine_list:
        
        if cuisine in user_question:
            for i in model.most_similar(cuisine):
                if i[0] not in cuisine_list:
                    ingredient_list.append(i[0])
            print("Bot: " + "\t" + "commonly used ingredients in {} food are {}".format(cuisine, ingredient_list)
                          + "\n" + "\t" + "Do you want to know more? yes or no")
                        
    if user_question == "":
        print("Bot: "+ "\t" + "What kind of cuisine do you want to know?"
                   + "\t" + "I can tell you common ingredients often used in the country." 
                   + "\n" + "\t" + "What kind of cuisine do you want to know?" 
                   + "\n" + "\t" + "Cuisine list is here."
                   + "\n" + "\t" + "====================[CUISINE LISE]===================="
                   + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
                   + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
                   + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
                   + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
                   + "\n" + "\t" + "======================================================")
        
    elif user_question == "yes":
        print("Bot: "+ "\t" + "What kind of cuisine do you want to know?"
                   + "\t" + "I can tell you common ingredients often used in the country." 
                   + "\n" + "\t" + "What kind of cuisine do you want to know?" 
                   + "\n" + "\t" + "Cuisine list is here."
                   + "\n" + "\t" + "====================[CUISINE LISE]===================="
                   + "\n" + "\t" + "italian, mexican, southern_us, indian, chinese"
                   + "\n" + "\t" + "french, cajun_creole, thai, japanese, greek"
                   + "\n" + "\t" + "spanish, korean, vietnamese', moroccan, british"
                   + "\n" + "\t" + "filipino, irish, jamaican, russian, brazilian"
                   + "\n" + "\t" + "======================================================")
                        
    elif user_question == "no":
        print("Bot: "+ "\t" + "See you again. Bye.")
        break






