from GenerateRandomState import *
from DataInitializer import *
from Classify import *
from PreProcessing import *
from PlotResult import *

import string

import numpy as np
import random 
import json
from os import path

CLF = None




def main():
    global CLF
    df = getDFFromXML("train_data.xml")
    print(df)

    df['text'] = df.text.apply(lambda text : preProcessing(text))
    print(df)
    print("\n\n#################################################################################################\n\n")
    #classification(df)

    dfTest = getDFFromXML("test_data.xml")
    print(dfTest)
    dfTest['text'] = dfTest.text.apply(lambda text : preProcessing(text))
    print(dfTest)

    print("\n\n#################################################################################################\n\n")
    #Classification(df)
    CLF = LaptopClassifier()
    CLF.CreateClassifieur(df,ALL_CATEGORIES)
    CLF.predictTestData(dfTest)

    

    print("\n\n#################################################################################################\n\n")
    CLF.ClassifyString("Battery could be better but it has robust processor and plenty of RAM so that is a trade off I suppose.")
    print("\n\n#################################################################################################\n\n")
    CLF.ClassifyString("slow processor, just not it")
    print("\n\n#################################################################################################\n\n")
    CLF.ClassifyString("Battery life is astonishing given the processing power and high resolution display.")
    print("\n\n#################################################################################################\n\n")
    CLF.ClassifyString("The processor shows a speed of 1.7gz.")
    print("\n\n#################################################################################################\n\n")
    CLF.ClassifyString( "Overall good but processing power isn't very good.")
    



if __name__=="__main__":
    main()






















# def classification(df):
#     df = df.drop(labels = ['id'], axis=1)#No use

#     vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
#     matrice = vectorizer.fit_transform(df.text)

#     #voir pour avoir le fichier de test
#     x_train = vectorizer.transform(df.text)
#     y_train = df.drop(labels = ['text'], axis=1)
    
#     # Using pipeline for applying logistic regression and one vs rest classifier
#     LogReg_pipeline = Pipeline([
#                 ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=1000), n_jobs=-1)),
#             ])

#     for category in ALL_CATEGORIES:
#         print('**Processing {} comments...**'.format(category))
    
#         # Training logistic regression model on train data
#         LogReg_pipeline.fit(x_train, y_train[category])
    
#         # calculating test accuracy
#         prediction = LogReg_pipeline.predict(x_train)
#         print('Test accuracy is {}'.format(accuracy_score(y_train[category], prediction)))
#         print(confusion_matrix(y_train[category],prediction))
#         print("\n")