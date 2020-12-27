import xml.etree.ElementTree as et
import pandas as pd 

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


def getGeneralPolarity(polarities):
    currentPol = "neutral"#For case where no opinions is present.     

    for polarity in polarities:
        if(currentPol == "neutral"): #Positive or negative win on neutral
            currentPol = polarity 

        #negative and positive opinion create a mixed opinion
        if((currentPol == "positive" and polarity == "negative") or (currentPol == "negative" and polarity == "positive")):
            currentPol = "mixed"

    return POLARITY_DIC[currentPol]