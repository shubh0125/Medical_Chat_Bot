import re # for regex
import random
#importing all necessary libraries for ML
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]   #feature columns
X = training[cols]
Y = training['prognosis']



#assigning values to strings
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

#splitting dataset into test and train data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# Decision Tree
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)



#storing the data from csv into dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def getInfo():
    # name=input("Name:")
    print("Your Name",end="->")
    name=input("")
    print("hello ",name)

def check_pattern(dis_list,inp):
    import re
    pred_list=[]
    ptr=0
    regexp = re.compile(inp)
    for item in dis_list:

        if regexp.search(item):
            pred_list.append(item)
            # return 1,item
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return ptr,item

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    dT_clf = DecisionTreeClassifier()
    dT_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return dT_clf.predict([input_vector])

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")

def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    # print(val)
    disease = le.inverse_transform(val[0])
    return disease

reduced_data = training.groupby(training['prognosis']).max()

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    # print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []


    # conf_inp=int()
    while True:

        print("Enter the symptom you are experiencing \n",end="->")
        disease_input = input("")
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
 
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter number of days.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])

                print(description_list[present_disease[0]])



            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)



    recurse(0, 1)






def prognosisPredictor():
  getSeverityDict()
  getDescription()
  getprecautionDict()
  # getInfo()
  tree_to_code(clf,cols)


# fn for splitting the body of text into its constituent sentences
def splitIntoSentences(text):
  sentences = re.split("[.!?]", text)
  try:
    sentences.remove("")
  except:
    pass
  return sentences

# fn for extracting the individual words from the sentence
def tokenizer(sentence):
  words = re.split("[\s,]", sentence)
  try:
    words.remove("")
  except:
    pass
  return words

# stemming the words
def stemmer(tokens):
  stem = []
  for token in tokens:
    if token.endswith("ies"):
      token = token.replace('ies','y',1)
      stem.append(token)
    elif token.endswith('s'):
      stem.append(token[0:-1])
    elif token.endswith("ed"):
      stem.append(token[0:-2])
    elif token.endswith("ing"):
      stem.append(token[0:-3])
    else:
      stem.append(token)
  return stem

# function that can clean up any body of text
def cleanTxt(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[;:!\\"\'()\[\]]',"", text)
    text = re.sub(r'(<br\s*/><br\s*/>)|(\-)|(\/)'," ", text)
    return text

# function for processing the chat on user side
def userChat(request):
  processedText = []
  reqSentences = splitIntoSentences(request)
  for i in range(0,len(reqSentences)):
    words = tokenizer(reqSentences[i])
    stemmedWords = stemmer(words)
    stemmedSentence = ' '.join(stemmedWords)
    processedText.append(stemmedSentence)
  return processedText

# function to determine the response from the chatbot
def interpretChat(processedText):
  reply = ''
  for sentence in processedText:
    rand = random.random()
    if re.search("[H|h]i|[H|h]ello|[H|h]ey",sentence) is not None:
      replies = {0:'Hello! How may I help you?',1:'Hi how may I assist you?',2:'How may I be of assistance?'}
      reply = replies[round(rand*1000)%3]
      print(reply)
    elif re.search("[A|a]ppointent|[D|d]octor|[M|m]eet",sentence) is not None:
      replies = {0:'Would you like to contact a clinic?',1:'May I help you arrange an appointment?'}
      reply = replies[round(rand*1000)%2]
      print(reply)
    elif re.search("[T|t]hank|[H|h]elpful",sentence) is not None:
      replies = {0:'My pleasure! :)',1:'Glad I could be of help!',2:'Welcome!'}
      reply = replies[round(rand*1000)%3]
      print(reply)
    elif re.search("[F|f]eeling|[U|u]nwell|[S|s]ick", sentence) is not None:
      prognosisPredictor()
    elif re.search("[B|b]ye|[S|s]ee you|[G|g]ood|[T|t]hanks", sentence) is not None:
      replies = {0:'Goodbye!',1:'See you!'}
      reply = replies[round(rand*1000)%2]
      print(reply)
      return False
    else:
      replies = {0:'Pardon, I could not get what you were saying',1:'Sorry, I could not understand what you were saying. Would you mind rephrasing'}
      reply = replies[round(rand*1000)%2]
      reply += '\n Try interacting with Hi, Hey, Hello.'
      print(reply)
  return True




# Running the code

continueChat = True
print('You are using the Chatbot... type something')
while continueChat:
  chatIn = input()
  processedIN = userChat(chatIn)
  continueChat = interpretChat(processedIN)
