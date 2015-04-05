# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 07:21:42 2015

@author: jeppley
"""

import pandas as pd
import os as os
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import ensemble
import numpy
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import  GridSearchCV
from sklearn import tree
import matplotlib.pyplot as plt


os.getcwd()
os.chdir('C:\Users\jeppley\Dropbox\yelp_dataset_challenge_academic_dataset')

user_merge = pd.read_csv('user_to_model.csv')
user_merge.head()
user_merge.columns



# including our functions from last week up here for use. 
def cleanup_data(df, cutoffPercent = .01):
    for col in df:
        sizes = df[col].value_counts(normalize = True)
        # get the names of the levels that make up less than 1% of the dataset
        values_to_delete = sizes[sizes<cutoffPercent].index
        df[col].ix[df[col].isin(values_to_delete)] = "Other"
    return df
#

def get_binary_values(data_frame):
    """encodes cateogrical features in Pandas.
    """
    all_columns = pandas.DataFrame( index = data_frame.index)
    for col in data_frame.columns:
        data = pandas.get_dummies(data_frame[col], prefix=col.encode('ascii', 'replace'))
        all_columns = pandas.concat([all_columns, data], axis=1)
    return all_columns
#
def find_zero_var(df):
    """finds columns in the dataframe with zero variance -- ie those
        with the same value in every observation.
    """   
    toKeep = []
    toDelete = []
    for col in df:
        if len(df[col].value_counts()) > 1:
            toKeep.append(col)
        else:
            toDelete.append(col)
        ##
    return {'toKeep':toKeep, 'toDelete':toDelete} 
##
def find_perfect_corr(df):
    """finds columns that are eother positively or negatively perfectly correlated (with correlations of +1 or -1), and creates a dict 
        that includes which columns to drop so that each remaining column
        is independent
    """  
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  numpy.tril(corrMatrix.values, k = -1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(numpy.round(corrMatrix[col],10)) == 1.00].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    toRemove = []
    for item in result:
        toRemove.append(item[1:(len(item)+1)])
 #   print toRemove
   # toRemove = sum(toRemove, [])
    return {'corrGroupings':result, 'toRemove':toRemove}
###
    
explanatory_features = [col for col in user_merge.columns if col not in ['user_id', 'is_elite2']]
explanatory_df = user_merge[explanatory_features]

explanatory_df.dropna(how='all', inplace = True) 
explanatory_colnames = explanatory_df.columns

print explanatory_colnames


response_series = user_merge.is_elite2
response_series.dropna(how='all', inplace = True) 

removed = response_series.index[~response_series.index.isin(explanatory_df.index)]
print removed

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

print string_features
print numeric_features


imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = numeric_features
explanatory_df.head()


no_variation = find_zero_var(explanatory_df)
print no_variation
#looks like there is nothing to delete
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
print no_correlation
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

explanatory_df.dtypes

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)
print explanatory_df.head()


# creating a random forest object.
## these are the default values of the classifier
rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, verbose=0, min_density=None, compute_importances=None)


# let's compute ROC AUC of the random forest. 
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc')

roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring='roc_auc')

## let's compare the mean ROC AUC
print roc_scores_rf.mean()
print roc_score_tree.mean()


trees_range = range(10, 50, 10) #see what accuracy is like
param_grid = dict(n_estimators = trees_range)#tuning parameters is number estimators

grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc') 
grid.fit(explanatory_df, response_series) # often will want to do this after night, and after feature selection 

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]


# Plot the results of the grid search
plt.figure()
plt.plot(trees_range, grid_mean_scores)


best_rf_tree_est = grid.best_estimator_
# how many trees did the best estiator have? 
print best_rf_tree_est.n_estimators
# how accurate was the best estimator?
print grid.best_score_
## did accuracy improve? 


user_merge.fans[user_merge.is_elite2==1].describe()
elites = user_merge[user_merge.is_elite2==1]

# density plot of user fans, looks like
elites.fans.plot(kind='density', xlim=(0,100))


#Creating success metric for elites with certain levels of fans
user_merge['Elite_Fans'] = 0
user_merge.head()
user_merge.Elite_Fans[user_merge.fans>20]=1
user_merge.Elite_Fans[user_merge.Elite_Fans==1]

user_merge['totcomp'] = user_merge['compliments_cool'] + user_merge['compliments_hot'] + user_merge['compliments_cute'] + user_merge['compliments_funny'] +user_merge['compliments_more'] +user_merge['compliments_note'] +user_merge['compliments_photos']+user_merge['compliments_plain'] +user_merge['compliments_profile'] +user_merge['compliments_writer'] +user_merge['compliments_list']

user_merge.head()

compliments = user_merge[['compliments_cool', 'compliments_hot', 'compliments_cute', 'compliments_funny', 'compliments_more', 'compliments_note', 'compliments_photos', 'compliments_plain','compliments_profile','compliments_writer','compliments_list', 'totcomp']]
compliments.head()

#Merging compliment calculations back to full dataset for modeling
user_mergenew = user_merge.join(compliments.div(compliments['totcomp'], axis=0), rsuffix='_perc')

user_mergenew.head()
user_mergenew.describe()

user_mergenew['Elite_Fans'].value_counts()


#subsetting to Elite members only
elites = user_mergenew[user_merge.is_elite2==1]

#Among elite members, how many have fans >20 and how many have fans <20
elites['Elite_Fans'].value_counts()

#recoding all nan values for compliments to 0
elites.fillna(0)




#Beginning modeling of what makes an influential Elite vs. a non-influential elite


elites.columns

explanatory_features = elites[['compliments_cool_perc',	 'compliments_hot_perc',	 'compliments_cute_perc',	 'compliments_funny_perc',	 'compliments_more_perc',	 'compliments_note_perc',	 'compliments_photos_perc',	 'compliments_plain_perc',	 'compliments_profile_perc',	 'compliments_writer_perc']]
explanatory_df = explanatory_features
explanatory_df = explanatory_df.fillna(0)


explanatory_df.dropna(how='all', inplace = True) 
explanatory_colnames = explanatory_df.columns

print explanatory_colnames

response_series = elites.Elite_Fans
#response_series.dropna(how='all', inplace = True) 

#removed = response_series.index[~response_series.index.isin(explanatory_df.index)]
#print removed

string_features = explanatory_df.ix[:, explanatory_df.dtypes == 'object']
numeric_features = explanatory_df.ix[:, explanatory_df.dtypes != 'object']

print string_features
print numeric_features


imputer_object = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer_object.fit(numeric_features)
numeric_features = pandas.DataFrame(imputer_object.transform(numeric_features), columns = numeric_features.columns)

## pulling together numeric and encoded data.
explanatory_df = numeric_features
explanatory_df.head()


no_variation = find_zero_var(explanatory_df)
print no_variation
#looks like there is nothing to delete
explanatory_df.drop(no_variation['toDelete'], inplace = True)

# deleting perfect correlation
no_correlation = find_perfect_corr(explanatory_df)
print no_correlation
explanatory_df.drop(no_correlation['toRemove'], 1, inplace = True)

explanatory_df.dtypes

# scaling data
scaler = preprocessing.StandardScaler()
scaler.fit(explanatory_df)
explanatory_df = pandas.DataFrame(scaler.transform(explanatory_df), columns = explanatory_df.columns)
print explanatory_df.head()


# creating a random forest object.
## these are the default values of the classifier
rf = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, random_state=None, verbose=0, min_density=None)


# let's compute ROC AUC of the random forest. 
roc_scores_rf = cross_val_score(rf, explanatory_df, response_series, cv=10, scoring='roc_auc')

roc_score_tree = cross_val_score(tree.DecisionTreeClassifier(), explanatory_df, response_series, cv=10, scoring='roc_auc')

## let's compare the mean ROC AUC
print roc_scores_rf.mean()
print roc_score_tree.mean()
#random forest is definitely pretty good here; 84% versus the 65% for a decision tree

trees_range = range(10, 100, 10) #see what accuracy is like
param_grid = dict(n_estimators = trees_range)#tuning parameters is number estimators

grid = GridSearchCV(rf, param_grid, cv=10, scoring='roc_auc') 
grid.fit(explanatory_df, response_series) # often will want to do this after night, and after feature selection 

# Check out the scores of the grid search
grid_mean_scores = [result[1] for result in grid.grid_scores_]


# Plot the results of the grid search
plt.figure()
plt.plot(trees_range, grid_mean_scores)
#looks like 80 is a good figure

best_rf_tree_est = grid.best_estimator_
# how many trees did the best estiator have? 
print best_rf_tree_est.n_estimators
# how accurate was the best estimator? .84
print grid.best_score_
## did accuracy improve? stayed about the same, so think 40 is really the magic number

rf.fit(explanatory_df, response_series)

important_features = rf.feature_importances_


#################
## BOOSTING TREES
#################
boosting_tree = ensemble.GradientBoostingClassifier()

roc_scores_gbm = cross_val_score(boosting_tree, explanatory_df, response_series, cv=10, scoring='roc_auc')

#let's compare our accuracies
print roc_scores_gbm.mean()
print roc_scores_rf.mean()
print roc_score_tree.mean()

# let's tune for num_trees, learning rate, and subsampling percent.
# need to import arange to create ranges for floats
from numpy import arange #pythons range function doesn't allow you to do floats

learning_rate_range = arange(0.01, 0.4, 0.02)
subsampling_range = arange(0.25, 1, 0.25)
n_estimators_range = range(25, 100, 25) #less than RF because by definition you are boosting

param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range, subsample = subsampling_range)

gbm_grid = GridSearchCV(boosting_tree, param_grid, cv=10, scoring='roc_auc')
gbm_grid.fit(explanatory_df, response_series)

# find the winning parameters
print gbm_grid.best_params_
# how does this compare to the default settings
# estimators = 100, subsample = 1.0, learning_rate = 0.1

# pull out the best score
print gbm_grid.best_score_
print grid.best_score_
## only slightly better than RF even after all the grid searching


## ROC curve accuracy of the GBM vs RF vs Tree Method

#not doing on all the CV splits
from sklearn.cross_validation import train_test_split
from sklearn import metrics

xTrain, xTest, yTrain, yTest = train_test_split(
                    explanatory_df, response_series, test_size =  0.3)


#comparing ROCs of our best estimators that came out of grid search
tree_probabilities = pandas.DataFrame(tree.DecisionTreeClassifier().fit(xTrain, yTrain).predict_proba(xTest))#wrap in data frame because 2 columns of probabilities, one for 0 class and 1 class, pandas data frame easy to extract
rf_probabilities = pandas.DataFrame(best_rf_tree_est.fit(xTrain, yTrain).predict_proba(xTest))
gbm_probabilities = pandas.DataFrame(gbm_grid.best_estimator_.fit(xTrain, yTrain).predict_proba(xTest))


tree_fpr, tree_tpr, thresholds = metrics.roc_curve(yTest, tree_probabilities[1])
rf_fpr, rf_tpr, thresholds = metrics.roc_curve(yTest, rf_probabilities[1])
gbm_fpr, gbm_tpr, thresholds = metrics.roc_curve(yTest, gbm_probabilities[1])


plt.figure()
plt.plot(tree_fpr, tree_tpr, color = 'g')
plt.plot(rf_fpr, rf_tpr, color = 'b')
plt.plot(gbm_fpr, gbm_tpr, color = 'r')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
## what does this tell us for this sample?
##either random forest or boosting is always to left and above decision tree so these are the better option
##false positive is the "at the expense of axis"



## create partial dependence plot on most important features for gbm.

importances = pandas.DataFrame(gbm_grid.best_estimator_.feature_importances_, index = explanatory_df.columns, columns =['importance'])
#notice these are not sorted
#we do not know if this in importance in that you will be in the hall of fame or not, just that it's important


importances.sort(columns = ['importance'], ascending = False, inplace = True)
print importances

from sklearn.ensemble.partial_dependence import plot_partial_dependence

features = [i for i, j in enumerate(explanatory_df.columns.tolist()) if j in importances.importance[0:3].index.tolist()]
#only take features of top 3 importances, iterate through feature names that only match the top 3 importances
#i tells you which index in this list, that particular name occured in; these features exist in indexes 2,5,16 of column names; j is the feature name
#iterating through two objects simultaneously,  loop through the index and value, and only pull out the value when the index is in the top 3 feature importances


fig, axs = plot_partial_dependence(gbm_grid.best_estimator_, explanatory_df, features, feature_names = explanatory_df.columns)
#look at a number of certain, where likelihood really rises a lot [this gives you a threshold], if this was not scaled axis










































