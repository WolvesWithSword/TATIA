import pandas as pd 
import string
import xml.etree.ElementTree as et
import numpy as np

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
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=1000, n_jobs=-1), n_jobs=-1)),
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
        report = classification_report(y_resample,prediction)
        print(report)


    #POLARITY
    print('**Processing polarity...**')
    LogReg_pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=1000 ,n_jobs=-1), n_jobs=-1)),
    ])
    Classifieur['polarity'] = LogReg_pipeline.fit(x_train, y_train['polarity'])
    prediction = Classifieur['polarity'].predict(x_train)
    print(classification_report(y_train['polarity'],prediction))

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
        print("Pour la categorie ",category," :")
        prediction = Classifieur[category].predict(x_test)

        print(classification_report(y_test[category],prediction))
        scores.append(np.mean(precision_score(y_test[category],prediction,average=None)))
        names.append(category)

    #polarity classification
    predictionPol = Classifieur['polarity'].predict(x_test)
    print(classification_report(y_test['polarity'],predictionPol))

    plt.bar(names, scores)
    plt.xticks(rotation="vertical")
    plt.show()







df = getDFFromXML("train_data.xml")
print(df)

df['text'] = df.text.apply(lambda text : preProcessing(text))
print(df)
print("\n\n#################################################################################################\n\n")
#classification(df)

Classifieur,vectorizer,ClassifyCategory = CreateClassifieur(df)

dfTest = getDFFromXML("test_data.xml")
predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest)
print("\n\n#################################################################################################\n\n")
ClassifyString(Classifieur,vectorizer,ClassifyCategory,"What a great laptop for its built quality and performance.")
print("\n\n#################################################################################################\n\n")
ClassifyString(Classifieur,vectorizer,ClassifyCategory,"Display is incredibly clear and fast.")
print("\n\n#################################################################################################\n\n")
ClassifyString(Classifieur,vectorizer,ClassifyCategory,"However, I would say the keyboard is extremely satisfying to type on and I sometimes even use it instead of my mechanical keyboard.")
print("\n\n#################################################################################################\n\n")
ClassifyString(Classifieur,vectorizer,ClassifyCategory,"This laptop will run the game on high settings, however the fans will be very loud and the CPU will be working very hard.")
print("\n\n#################################################################################################\n\n")
ClassifyString(Classifieur,vectorizer,ClassifyCategory,"As a stand alone laptop it is outstanding, especially at the $700 price point.")























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