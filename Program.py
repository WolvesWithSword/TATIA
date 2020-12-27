from GenerateRandomState import *
from DataInitialiser import *
from Classify import *
from PreProcessing import *
from PlotResult import *

import string

import numpy as np
import random 
import json
from os import path


def ClassifyString(Classifieur,vectorizer,ClassifyCategory,string):
    print(string, " -> ")
    preComputeString = vectorizer.transform([preProcessing(string)])

    for category in ClassifyCategory:
        
        prediction = Classifieur[category].predict(preComputeString)
        if(prediction[0]==1):
            print(category)
    
    #POLARITY
    predictionPol = Classifieur['polarity'].predict(preComputeString)
    print("The polarity is :",POLARITY_DIC_ANSWER[predictionPol[0]])

def main():

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
    Classifieur,vectorizer,ClassifyCategory = CreateClassifieur(df,ALL_CATEGORIES)
    predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest)


    print("\n\n#################################################################################################\n\n")

    '''
    if(path.exists(FILE_RANDOM_STATE)):
        f = open(FILE_RANDOM_STATE, "r")
        randomStateDict=json.load(f)

    categoryToGenerate= ALL_CATEGORIES[:]
    categoryToGenerate.append("polarity")
    randomStateDict=generateRandomStateDictionary(df,categoryToGenerate,randomState=randomStateDict)

    Classifieur,vectorizer,ClassifyCategory = CreateClassifieurWithRandomState(df,randomStateDict)
    predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest)
    '''
    

    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"W7 Pro with W8 pro upgrade is nice, but it frequently freezes for a few seconds here and there.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"It took some getting used to on the Widows 8.1, after being an XP user for many years.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Other than a hate of windows 8, she just loves it.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"So with this new order, all the upgrades and windows 8 really caught my eye and I got this for the long haul.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"The Windows 8.1 used to boot in just 2s and that was freaking awesome.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"I liked the fact that it came with Windows 7, but also included Windows 8 to install later if I need it.")
    # print("\n\n#################################################################################################\n\n")
    # ClassifyString(Classifieur,vectorizer,ClassifyCategory,"The 8.1 windows is a major disappointment.")



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