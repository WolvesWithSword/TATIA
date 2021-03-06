from PlotResult import *
from PreProcessing import preProcessing
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

class LaptopClassifier:

    def CreateClassifieur(self,df,allCategory):
        df = df.drop(labels = ['id'], axis=1) #No Use

        self.vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,8), norm='l2')
        matrice = self.vectorizer.fit_transform(df.text)

        #voir pour avoir le fichier de test
        x_train = self.vectorizer.transform(df.text)
        y_train = df.drop(labels = ['text'], axis=1)
        
        self.classifieur = dict()

        self.classifyCategory = []
    
        for category in allCategory:

            #INCREASE POPULATION 
            ADA = ADASYN(sampling_strategy='auto',n_neighbors = 4)
        
            nb_class = getNumberOfClass(y_train[category])
            if(nb_class > 1): #can't resample if there is one class.
                x_resample, y_resample =  ADA.fit_resample(x_train,y_train[category])
                self.classifyCategory.append(category)
            else :
                continue

            print('**Processing {} comments...**'.format(category))
            # Using pipeline for applying logistic regression and one vs rest classifier
            LogReg_pipeline = Pipeline([
                    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=1000, n_jobs=-1), n_jobs=-1)),
                ])

            # Training logistic regression model on train data
            clf = LogReg_pipeline.fit(x_resample, y_resample)

            self.classifieur[category] = clf
            prediction = clf.predict(x_train)
            report = classification_report(y_train[category],prediction,zero_division=1)
            print(report)

        #INCREASE POPULATION 
        ADAPol = ADASYN(sampling_strategy='minority',n_neighbors = 10)
        x_resample, y_resample =  ADAPol.fit_resample(x_train,y_train['polarity'])

        #POLARITY
        print('**Processing polarity...**')
        LogReg_pipeline = Pipeline([
            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', class_weight='balanced', max_iter=1000 ,n_jobs=-1), n_jobs=-1)),
        ])
        self.classifieur['polarity'] = LogReg_pipeline.fit(x_resample, y_resample)

        prediction = self.classifieur['polarity'].predict(x_train)
        print(classification_report(y_train['polarity'],prediction,zero_division=1))


        return self

    def orArray(self,L1,L2):
        for i in range(len(L1)):
            L1[i] = L1[i] | L2[i]

    def predictTestData(self,dfTest):
        dfTest = dfTest.drop(labels = ['id'], axis=1) #No Use

        #voir pour avoir le fichier de test
        x_test = self.vectorizer.transform(dfTest.text)
        y_test = dfTest.drop(labels = ['text'], axis=1)

        names=[]
        f1scoresFalse = []
        f1scoresTrue = []
        scoresTrue = []
        supportTrue = []
        predictEntity = dict()
        realEntity = dict()
        for category in self.classifyCategory:
            
            print("Pour la categorie",category,":")
            prediction = self.classifieur[category].predict(x_test)

            print(classification_report(y_test[category],prediction,zero_division=1))
            
            #DATA FOR PLOT
            entity = category.split("#")[0]
            if(not entity in predictEntity):
                predictEntity[entity] = prediction[:]
                realEntity[entity] = y_test[category][:]
            else:
                self.orArray(predictEntity[entity],prediction)
                self.orArray(realEntity[entity],y_test[category])

            if(sum(y_test[category]) == 0) : 
                continue
            
            report = classification_report(y_test[category],prediction,zero_division=1, output_dict=True)
            supportTrue.append(report['1']['support'])
            scoresTrue.append(report['1']['precision'])
            f1scoresFalse.append(report['0']['f1-score'])
            f1scoresTrue.append(report['1']['f1-score'])
            #scores.append(np.mean(precision_score(y_test[category],prediction,average=None)))
            names.append(category)

        #polarity classification
        print("Pour la polarité :")
        predictionPol = self.classifieur['polarity'].predict(x_test)
        print(classification_report(y_test['polarity'],predictionPol,zero_division=1))

        plotTrueData(names,scoresTrue,supportTrue)
        plotDataF1Score(names, f1scoresFalse, f1scoresTrue)
        plotDataEntity(predictEntity,realEntity)
        plotPolarity(predictionPol,y_test['polarity'])

    def ClassifyString(self,string):
        if(self.classifieur==None):
            print("not initialised clf")
            return
        print(string, " -> ")
        preComputeString = self.vectorizer.transform([preProcessing(string)])

        for category in self.classifyCategory:
            
            prediction = self.classifieur[category].predict(preComputeString)
            if(prediction[0]==1):
                print(category)
        
        #POLARITY
        predictionPol = self.classifieur['polarity'].predict(preComputeString)
        print("The polarity is :",POLARITY_DIC_ANSWER[predictionPol[0]])

def getNumberOfClass(y_tab):
    class_list = []
    nb = 0

    for c_class in y_tab :
        if(c_class not in class_list):
            nb += 1
            class_list.append(c_class)
        
    return nb