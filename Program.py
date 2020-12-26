import pandas as pd 
import string
import xml.etree.ElementTree as et
import numpy as np
import random 
import json
from os import path

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN

# nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#graphique
import matplotlib.pyplot as plt

FILE_RANDOM_STATE = "randomState.json"

PONCTUATION = set(string.punctuation)
STOP_WORDS = stopwords.words("english")

STOP_WORDS.extend(PONCTUATION)


ENTITIES = ["LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD",
"CPU", "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY",
"OPTICAL_DRIVES", "BATTERY", "GRAPHICS", "HARD_DISK",
"MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS",
"WARRANTY", "SHIPPING", "SUPPORT", "COMPANY"]

ATTRIBUTES = ["GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES",
"OPERATION_PERFORMANCE", "USABILITY", "PORTABILITY",
"CONNECTIVITY", "MISCELLANEOUS"]

POLARITY_DIC = {"negative" : 0, "positive":1, "neutral":2, "mixed":3}
POLARITY_DIC_ANSWER = {0 :"negative" , 1:"positive", 2:"neutral", 3:"mixed"}

def allCategoryClass(entities,attributes):
    all_cat = []

    for entity in entities:
        for attribute in attributes:
            all_cat.append(entity+"#"+attribute)

    return all_cat


ALL_CATEGORIES  = allCategoryClass(ENTITIES,ATTRIBUTES)



def getDFFromXML(fileName): #WORK ONLY FOR OUR XML TRAIN FILE
    xtree = et.parse(fileName)
    x_reviews = xtree.getroot()

    df_cols = ["id", "text","polarity"]
    for category in ALL_CATEGORIES:
        df_cols.append(category)

    rows = []

    for node_review in x_reviews: 
        for node_sentences in node_review:
            for node_sentence in node_sentences:
                s_id = node_sentence.attrib.get("id")
                s_text = node_sentence.find("text").text if node_sentence is not None else None

                category_list = []
                polarity_list =[]

                for node_opinions in node_sentence:
                    for node_opinion in node_opinions:
                        category_list.append(node_opinion.attrib.get("category"))
                        polarity_list.append(node_opinion.attrib.get("polarity"))

                binary_category_tab = binaryCategoryTab(ALL_CATEGORIES,category_list)

                #CONSTRUCTION DU DICO
                dictionary = {}
                dictionary["id"] = s_id
                dictionary["text"] = s_text
                dictionary["polarity"] = getGeneralPolarity(polarity_list)#Convert into int

                for i in range(len(ALL_CATEGORIES)):
                    dictionary[ALL_CATEGORIES[i]] = binary_category_tab[i]

                rows.append(dictionary)

    out_df = pd.DataFrame(rows, columns = df_cols)
    return out_df

def binaryCategoryTab(all_categories,current_categories):
    res = []

    for category in all_categories:
        if(category in current_categories):
            res.append(1)
        else:
            res.append(0)
    
    return res

def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens_without_stopwords = [word for word in tokens if word not in STOP_WORDS]
    return tokens_without_stopwords

def lemmatize(tokens):
    lemma = WordNetLemmatizer()
    tokens_lemmatize = [lemma.lemmatize(word,'a') for word in tokens]
    tokens_lemmatize = [lemma.lemmatize(word,'v') for word in tokens]
    tokens_lemmatize = [lemma.lemmatize(word,'n') for word in tokens]
    return tokens_lemmatize

def preProcessing(text):
    tokens = tokenize(text)
    tokens_lemmatize = lemmatize(tokens)
    return " ".join(tokens_lemmatize)


def CreateClassifieur(df):
    df = df.drop(labels = ['id'], axis=1) #No Use

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
    matrice = vectorizer.fit_transform(df.text)

    #voir pour avoir le fichier de test
    x_train = vectorizer.transform(df.text)
    y_train = df.drop(labels = ['text'], axis=1)
    
    Classifieur = dict()

    ClassifyCategory = []
   
    for category in ALL_CATEGORIES:

        
        #INCREASE POPULATION 
        ADA = ADASYN(sampling_strategy='auto',n_neighbors = 4)
    
        nb_class = getNumberOfClass(y_train[category])
        if(nb_class > 1): #can't resample if there is one class.
            x_resample, y_resample =  ADA.fit_resample(x_train,y_train[category])
            ClassifyCategory.append(category)
        else :
            continue
        
        x_resample, y_resample = x_train,y_train[category]

        correct = False
        bestClf = None
        bestClfPrecision = 0
        nbExec = 0
        while(not correct):
            print('**Processing {} comments...**'.format(category))
            # Using pipeline for applying logistic regression and one vs rest classifier
            LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=500, n_jobs=-1), n_jobs=-1)),
                ])

            # Training logistic regression model on train data
            currentClf = LogReg_pipeline.fit(x_resample, y_resample)
    
            prediction = currentClf.predict(x_resample)

            #si incorrect on refait le train
            precision_Score = precision_score(y_resample, prediction, average=None)
            if(len(precision_Score) == 2):

                #on a pas de clasifieur on l'ajoute
                if(bestClf==None or precision_Score[1]>bestClfPrecision):
                    bestClf = currentClf
                    bestClfPrecision = precision_Score[1]

                #On relance si la precision n'est pas suffisante (pour nous)
                if(precision_Score[1]<0.8 and nbExec<10):
                    nbExec+=1
                    continue
            else: #si on a pas de classification c'est le meillieur
                bestClf = currentClf

            #on garde la classification
            correct=True

        Classifieur[category] = bestClf
        prediction = bestClf.predict(x_resample)
        report = classification_report(y_resample,prediction,zero_division=1)
        print(report)


    #POLARITY
    print('**Processing polarity...**')
    LogReg_pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=500 ,n_jobs=-1), n_jobs=-1)),
    ])
    Classifieur['polarity'] = LogReg_pipeline.fit(x_train, y_train['polarity'])
    prediction = Classifieur['polarity'].predict(x_train)
    print(classification_report(y_train['polarity'],prediction,zero_division=1))

    return Classifieur,vectorizer,ClassifyCategory

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

def getNumberOfClass(y_tab):
    class_list = []
    nb = 0

    for c_class in y_tab :
        if(c_class not in class_list):
            nb += 1
            class_list.append(c_class)
    
    return nb

def getGeneralPolarity(polarities):
    currentPol = "neutral"#For case where no opinions is present.     

    for polarity in polarities:
        if(currentPol == "neutral"): #Positive or negative win on neutral
            currentPol = polarity 

        #negative and positive opinion create a mixed opinion
        if((currentPol == "positive" and polarity == "negative") or (currentPol == "negative" and polarity == "positive")):
            currentPol = "mixed"

    return POLARITY_DIC[currentPol]


def predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest):
    dfTest = dfTest.drop(labels = ['id'], axis=1) #No Use

    #voir pour avoir le fichier de test
    x_test = vectorizer.transform(dfTest.text)
    y_test = dfTest.drop(labels = ['text'], axis=1)

    names=[]
    scores = []
    for category in ClassifyCategory:
        if(sum(y_test[category]) == 0) : 
            continue

        print("Pour la categorie",category,":")
        prediction = Classifieur[category].predict(x_test)

        print(classification_report(y_test[category],prediction,zero_division=1))
        scores.append(np.mean(precision_score(y_test[category],prediction,average=None)))
        names.append(category)

    #polarity classification
    predictionPol = Classifieur['polarity'].predict(x_test)
    print(classification_report(y_test['polarity'],predictionPol,zero_division=1))

    plt.bar(names, scores)
    plt.xticks(rotation="vertical")
    plt.show()






def generateRandomStateDictionary(df,categoryToGenerate,maxExec=100,maxRandomState=5000,
    randomState=dict(),writeInFile=True,fileName=FILE_RANDOM_STATE ,upgradeMode=True):

    if(maxExec>=maxRandomState/10):
        raise Exception("generateRandomStateDictionary: value not accepted -> Max exec >= maxRandomeState/10")

    df = df.drop(labels = ['id'], axis=1) #No Use

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
    matrice = vectorizer.fit_transform(df.text)

    #voir pour avoir le fichier de test
    x_train = vectorizer.transform(df.text)
    y_train = df.drop(labels = ['text'], axis=1)
    progress = 0
    for category in categoryToGenerate:
        
        print("Generate random state progress : ",progress/len(categoryToGenerate)*100,"%")

        nb_class = getNumberOfClass(y_train[category])
        # il n'y a pas plusieur categorie on ne peut pas trouver un bon randome state
        if(nb_class <= 1):
            randomState[category]=-1
            continue

        x_resample, y_resample = x_train,y_train[category]

        bestScore = 0
        bestRandomState = 0
        alreadyTestRandom = dict()
        nbExec = 0
        for nbExec in range(maxExec):
            #premiere execution on recuprer un element deja existant si possible (si l'on veut ameillioré un resultat deja existant)
            if(nbExec==0 and upgradeMode and category in randomState and randomState[category]!=-1):
                currentRandom = randomState[category]
            else:
                #generation d'un random non utilisé
                currentRandom = random.randint(1,maxRandomState)
                while (currentRandom in alreadyTestRandom):
                    currentRandom = random.randint(1,maxRandomState)

            # Using pipeline for applying logistic regression and one vs rest classifier
            LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced',random_state=currentRandom, max_iter=1000, n_jobs=-1), n_jobs=-1)),
                ])

            # Training logistic regression model on train data
            currentClf = LogReg_pipeline.fit(x_resample, y_resample)
    
            prediction = currentClf.predict(x_resample)

            #si incorrect on refait le train
            accuracy = accuracy_score(y_resample, prediction)

            if(accuracy>bestScore):
                bestScore=accuracy
                bestRandomState = currentRandom
                #Best score
                if(accuracy==1):
                    break



        randomState[category]=bestRandomState
        progress+=1

    print(randomState)

    #write on file
    with open(fileName, 'w') as file:
     file.write(json.dumps(randomState)) # use `json.loads` to do the reverse

    return randomState

def CreateClassifieurWithRandomState(df,randomStateDict):
    df = df.drop(labels = ['id'], axis=1) #No Use

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
    matrice = vectorizer.fit_transform(df.text)

    #voir pour avoir le fichier de test
    x_train = vectorizer.transform(df.text)
    y_train = df.drop(labels = ['text'], axis=1)
    
    Classifieur = dict()

    ClassifyCategory = []
   
    for category in ALL_CATEGORIES:

        
        #INCREASE POPULATION 
        #ADA = ADASYN(sampling_strategy='auto',n_neighbors = 4)
    
        nb_class = getNumberOfClass(y_train[category])
        if(nb_class > 1): #can't resample if there is one class.
            #x_resample, y_resample =  ADA.fit_resample(x_train,y_train[category])
            ClassifyCategory.append(category)
        else :
            continue
        
        x_resample, y_resample = x_train,y_train[category]

        print('**Processing {} comments...**'.format(category))
        # Using pipeline for applying logistic regression and one vs rest classifier
        LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced',random_state=randomStateDict[category] ,max_iter=1000, n_jobs=-1), n_jobs=-1)),
            ])

        # Training logistic regression model on train data
        Clf = LogReg_pipeline.fit(x_resample, y_resample)


        Classifieur[category] = Clf
        prediction = Clf.predict(x_resample)
        report = classification_report(y_resample,prediction,zero_division=1)
        print(report)


    #POLARITY
    print('**Processing polarity...**')
    LogReg_pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced',random_state=randomStateDict["polarity"] ,max_iter=1000 ,n_jobs=-1), n_jobs=-1)),
    ])
    Classifieur['polarity'] = LogReg_pipeline.fit(x_train, y_train['polarity'])
    prediction = Classifieur['polarity'].predict(x_train)
    print(classification_report(y_train['polarity'],prediction,zero_division=1))

    return Classifieur,vectorizer,ClassifyCategory


def computeRandomStateArray(df):
    if(path.exists(FILE_RANDOM_STATE)):
        f = open(FILE_RANDOM_STATE, "r")
        try:
            randomStateDict=json.load(f)
            print(randomStateDict)
        except:
            randomStateDict=dict()
    else:
        randomStateDict=dict()

    #permet de compute pour tout les categorie (stock dans un fichier donc en commentaire mtn)
    CatRandomState = ALL_CATEGORIES[:]
    CatRandomState.append("polarity")
    generateRandomStateDictionary(df,CatRandomState,randomState=randomStateDict,maxExec=100)

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
    #classification(df)
    # Classifieur,vectorizer,ClassifyCategory = CreateClassifieur(df)
    #predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest)

    if(path.exists(FILE_RANDOM_STATE)):
        f = open(FILE_RANDOM_STATE, "r")
        randomStateDict=json.load(f)

    Classifieur,vectorizer,ClassifyCategory = CreateClassifieurWithRandomState(df,randomStateDict)
    predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest)

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