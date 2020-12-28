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
    

    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Battery could be better but it has robust processor and plenty of RAM so that is a trade off I suppose.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"slow processor, just not it")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Battery life is astonishing given the processing power and high resolution display.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"The processor shows a speed of 1.7gz.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Overall good but processing power isn't very good.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"And processor feels pressured even though I run medium scale programs, watching the way exhaust speed increases.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Pretty much every major game save for Solitare won't even play on it due to the slower processor.")

    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Performance wise, it's a bit slow, but that's to be expected from a Celeron-class processor.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Other than the slow CPU it works great for everyday use.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"The i7 is incredibly fast, the RAM makes it proper for several games (not on maximum quality).")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Retina screen, solid build quality, weight, fast processor and the reliability of the Apple brand.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"The laptop is very responsice with the Core i7 processor and 8GB of RAM and the sound out of the Beats audio is great.")
    print("\n\n#################################################################################################\n\n")
    ClassifyString(Classifieur,vectorizer,ClassifyCategory,"")




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