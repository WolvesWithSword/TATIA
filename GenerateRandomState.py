from Program import *
FILE_RANDOM_STATE = "randomState.json"

def generateRandomStateDictionary(df,categoryToGenerate,maxExec=100,maxRandomState=100000,
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
            progress+=1
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
            score = precision_score(y_resample, prediction)

            if(score>bestScore):
                bestScore=score
                bestRandomState = currentRandom
                #Best score
                if(score==1):
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