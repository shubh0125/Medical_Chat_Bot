import re # for regex
import random
#importing all necessary libraries for ML
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split

#ML part
training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')
cols= training.columns
cols= cols[:-1]   #feature columns
X = training[cols]
Y = training['prognosis']

def getSymptoms():
  global symptomList
  symptomList = list(training.columns)

#assigning values to strings
le = preprocessing.LabelEncoder()
le.fit(Y)
Y = le.transform(Y)

#splitting dataset into test and train data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Training Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(x_train, y_train)
pred_y = rf_model.predict(x_test)

def prognosisPredictor():
  getSymptoms()
  print('Enter your symptoms:')
  sym = input()
  symptoms = tokenizer(sym)
  inputVector = 132*[0]
  for symptom in symptoms:
    try:
      x = symptomList.index(symptom)
      inputVector[x] = 1
    except:
      pass
  #print(inputVector)
  pred = rf_model.predict([inputVector])
  print('You may be suffering from: ', le.inverse_transform(pred)[0])


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
  continueChat = True
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
    elif re.search("[F|f]eeling|[U|u]nwell|[S|s]ick", sentence) is not None:
      prognosisPredictor()
    elif re.search("[B|b]ye|[S|s]ee you|[G|g]ood", sentence) is not None:
      replies = {0:'Goodbye!',1:'See you!'}
      reply = replies[round(rand*1000)%2]
      print(reply)
      return False
  return True


continueChat = True
print('You are using the Chatbot... type something')
while continueChat:
  chatIn = input()
  processedIN = userChat(chatIn)
  continueChat = interpretChat(processedIN)
