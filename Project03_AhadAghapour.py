'''
===================================================
Created on Apr 1, 2017
Project03 of Data Science
@author: Ahad Aghapour
@student number: S011178
@email: ahad.aghapour@ozu.edu.tr
===================================================
'''


import os
import numpy as np 
import scipy as sc
import pandas as pd
from sklearn import svm, model_selection, feature_extraction, linear_model, tree, ensemble, neural_network
from conda_build.create_test import header



def eval(cls,k,data,target, method=None, test_ratio=0.25):
    score = 0
    if method == 'kfold':
        kfold = model_selection.KFold(n_splits=k)
        for ind_train,ind_test in kfold.split(data):
            cls.fit(data[ind_train], target[ind_train])
            ypred = cls.predict(data[ind_test])
            score += cls.score(data[ind_test],target[ind_test]) #np.mean((ypred == iris.target[ind_test]).astype(int))
    else:
        for I in range(k):
            xtrain,xtest,ytrain,ytest=model_selection.train_test_split(data,target,test_size=test_ratio)
            cls.fit(xtrain,ytrain)
            score += cls.score(xtest,ytest) 
    return score/k


# in this approach we repeate the train data 3 time
def eval2(cls,k,data,target, method=None, test_ratio=0.25):
    score = 0
    if method == 'kfold':
        kfold = model_selection.KFold(n_splits=k)
        for ind_train,ind_test in kfold.split(data):
            cls.fit(data[ind_train], target[ind_train])
            ypred = cls.predict(data[ind_test])
            score += cls.score(data[ind_test],target[ind_test]) #np.mean((ypred == iris.target[ind_test]).astype(int))
    else:
        for I in range(k):
            xtrain,xtest,ytrain,ytest=model_selection.train_test_split(data,target,test_size=test_ratio)
            
            xtrain=np.concatenate((xtrain.toarray(),xtrain.toarray(), xtrain.toarray(), xtrain.toarray()), axis=0)            
            ytrain=np.concatenate((ytrain, ytrain, ytrain, ytrain), axis=0)
              
            cls.fit(xtrain,ytrain)
            score += cls.score(xtest,ytest) 
    return score/k

# in this approach we repeate the train data 3 time and using the out train test function to do this randomly
def eval3(cls,k,data,target, method=None, test_ratio=0.25):
    score = 0
    for I in range(k):
        ind = np.random.permutation(data.shape[0])
        xtrain = data[ind[:int(test_ratio*data.shape[0])]]
        ytrain = target[ind[:int(test_ratio*data.shape[0])]]
        xtest = data[ind[int(test_ratio*data.shape[0]):]]
        ytest = target[ind[int(test_ratio*data.shape[0]):]]
        print('xtrain.shape: ',xtrain.shape)
        # duplicate train dataset
        xtrain = sc.concatenate((xtrain.toarray(), xtrain.toarray(), xtrain.toarray()))
        ytrain = sc.concatenate((ytrain, ytrain, ytrain))
        print('3xtrain.shape ',xtrain.shape)
        cls.fit(xtrain,ytrain)
        score += cls.score(xtest,ytest) 
    return score/k  


print(__doc__)




# load first dataset from file beside the python file
fileAddress=os.path.join(os.getcwd(), 'twitter_sentiment_corpus.csv')
print('Start to loading first data from this file: \n', fileAddress)

tweetsData = pd.read_csv(fileAddress)


# drop irrelevant target from our dataset
tweetsData = tweetsData.drop(tweetsData[tweetsData['Sentiment'] == 'irrelevant'].index)
 
  
# change target names to -1 0 1
row_index = tweetsData[tweetsData['Sentiment'] == 'positive'].index
tweetsData.loc[row_index, 'Sentiment'] = 1 
row_index = tweetsData[tweetsData['Sentiment'] == 'neutral'].index
tweetsData.loc[row_index, 'Sentiment'] = 0 
row_index = tweetsData[tweetsData['Sentiment'] == 'negative'].index
tweetsData.loc[row_index, 'Sentiment'] = -1


# # use the TweetId in TweetText
# tweetsData.TweetText = tweetsData.TweetId.map(str) + ' ' + tweetsData.TweetText


# separate the tweetData and tweetTarget
tweetData = np.array(tweetsData['TweetText'])
tweetTarget = np.array(tweetsData['Sentiment'].values, dtype="|S6")


stopWords = ['and', 'in', 'that', 'the', 'this']
# stopWords = None

##### Vectorizer
# 0    TfidfVectorizer
tfid=feature_extraction.text.TfidfVectorizer(use_idf=True,sublinear_tf=True, stop_words=stopWords)
newData=tfid.fit_transform(tweetData)
# 1    bigram TfidfVectorizer
tfid2=feature_extraction.text.TfidfVectorizer(use_idf=True, sublinear_tf=True, ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, stop_words=stopWords)
newData1=tfid2.fit_transform(tweetData)

# 2    CountVectorizer
countVectorize=feature_extraction.text.CountVectorizer(stop_words=stopWords)
newData2=countVectorize.fit_transform(tweetData)

# 3    bigram vectorizer
bigram_vectorizer = feature_extraction.text.CountVectorizer(ngram_range=(1, 2), min_df=1, stop_words=stopWords)
newData3 = bigram_vectorizer.fit_transform(tweetData)

# 4 HashingVectorizer
hashingVectorizer = feature_extraction.text.HashingVectorizer(n_features=100)
newData4 = hashingVectorizer.fit_transform(tweetData)

# Train iteration
maxAccuracy=0
for i in range(1):
    ###### Train and show the results
    
    print("\niteration {0} =======================================".format(i+1))
    print('\n1# TfidfVectorizer')
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData, target=tweetTarget)     
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression kfold:\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC kfold:\t\t{0:.4f}'.format(accuracy))    
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier kfold:\t\t{0:.4f}'.format(accuracy))    
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier kfold:\t\t{0:.4f}'.format(accuracy))    
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(100, 100), solver='sgd', max_iter=50, verbose=True), k=10, data=newData, target=tweetTarget)
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier:\t\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(100, 100), solver='sgd', verbose=True), k=10, data=newData, target=tweetTarget, method='kfold')
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    
    
    
    print('\n2# bigram_TfidfVectorizer')
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData1, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData1, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression kfold:\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData1, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData1, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData1, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData1, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData1, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData1, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier kfold:\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData1, target=tweetTarget)
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier:\t\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData1, target=tweetTarget, method='kfold')
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    
    
    print('\n3# CountVectorizer')
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData2, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData2, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression kfold:\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData2, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData2, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData2, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData2, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData2, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData2, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier kfold:\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData2, target=tweetTarget)
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier:\t\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData2, target=tweetTarget, method='kfold')
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    
    
    print('\n4# bigram_CountVectroizer')
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData3, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData3, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression kfold:\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData3, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData3, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData3, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData3, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData3, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData3, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier kfold:\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData3, target=tweetTarget)
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier:\t\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData3, target=tweetTarget, method='kfold')
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    
    print('\n5# HashingVectorizer')
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData4, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=linear_model.LogisticRegression(),k=10, data=newData4, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LogisticRegression kfold:\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData4, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=svm.LinearSVC(), k=10, data=newData4, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('LinearSVC kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData4, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=tree.DecisionTreeClassifier(max_depth=10), k=10, data=newData4, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('DecisionTreeClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData4, target=tweetTarget)
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier:\t\t\t{0:.4f}'.format(accuracy))
    accuracy = eval(cls=ensemble.RandomForestClassifier(), k=10, data=newData4, target=tweetTarget, method='kfold')
    maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
    print('RandomForestClassifier kfold:\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData4, target=tweetTarget)
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier:\t\t\t{0:.4f}'.format(accuracy))
#     accuracy = eval(cls=neural_network.MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', max_iter=50, verbose=True), k=10, data=newData4, target=tweetTarget, method='kfold')
#     maxAccuracy=accuracy if accuracy>maxAccuracy else maxAccuracy
#     print('MLPClassifier kfold:\t\t{0:.4f}'.format(accuracy))
    
    
    
print("\n===================================================")      
print("The maximum accuracy is: \t{0:.4f}".format(maxAccuracy))

