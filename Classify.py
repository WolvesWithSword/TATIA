from PlotResult import *
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


def CreateClassifieur(df,allCategory):
    df = df.drop(labels = ['id'], axis=1) #No Use

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
    matrice = vectorizer.fit_transform(df.text)

    #voir pour avoir le fichier de test
    x_train = vectorizer.transform(df.text)
    y_train = df.drop(labels = ['text'], axis=1)
    
    Classifieur = dict()

    ClassifyCategory = []
   
    for category in allCategory:

        
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




def predictTestData(Classifieur,vectorizer,ClassifyCategory,dfTest):
    dfTest = dfTest.drop(labels = ['id'], axis=1) #No Use

    #voir pour avoir le fichier de test
    x_test = vectorizer.transform(dfTest.text)
    y_test = dfTest.drop(labels = ['text'], axis=1)

    names=[]
    scores = []
    support = []
    for category in ClassifyCategory:
        if(sum(y_test[category]) == 0) : 
            continue

        print("Pour la categorie",category,":")
        prediction = Classifieur[category].predict(x_test)

        print(classification_report(y_test[category],prediction,zero_division=1))
        report = classification_report(y_test[category],prediction,zero_division=1, output_dict=True)
        support.append(report['1']['support'])
        scores.append(report['1']['precision'])
        #scores.append(np.mean(precision_score(y_test[category],prediction,average=None)))
        names.append(category)

    #polarity classification
    print("Pour la polarité :")
    predictionPol = Classifieur['polarity'].predict(x_test)
    print(classification_report(y_test['polarity'],predictionPol,zero_division=1))

    plotData(names,scores,support)



def getNumberOfClass(y_tab):
    class_list = []
    nb = 0

    for c_class in y_tab :
        if(c_class not in class_list):
            nb += 1
            class_list.append(c_class)
    
    return nb