import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fastapi import  FastAPI, Body, Request, Form, File 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="htmlFiles")

# In[4]:
@app.get("/{user_name}", response_class=HTMLResponse)
def runhome(request : Request, user_name:str):
 return templates.TemplateResponse("index.html", {"request":request, "username": user_name})


# # In[ ]:
@app.post("/checknews", response_class=HTMLResponse)
async def handleform(request : Request, newstring: str = Form(...)):
 df = pd.read_csv('news.csv')
 df.shape
 df.head()
#DataFlair - Get the labels
 labels=df.label
 labels.head()
# #DataFlair - Split the dataset
 x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
# #DataFlair - Initialize a TfidfVectorizer
 tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
 tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
 tfidf_test=tfidf_vectorizer.transform(x_test)

# #DataFlair - Initialize a PassiveAggressiveClassifier
 pac=PassiveAggressiveClassifier(max_iter=50)
 pac.fit(tfidf_train,y_train)

# #DataFlair - Predict on the test set and calculate accuracy
 y_pred=pac.predict(tfidf_test)
 score=accuracy_score(y_test,y_pred)
 print(f'Accuracy: {round(score*100,2)}%')


# #DataFlair - Build confusion matrix
 confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

 s = newstring


# # In[14]:


 matrix1 = tfidf_vectorizer.transform([s])
 predict = pac.predict(matrix1)



 print(predict[0])
    # newsmatrix : tfidf_vectorizer.transform([newstring])
    # predictnews : pac.predict(newsmatrix)
    # newsmatrix
 return templates.TemplateResponse("prediction.html", 
 {"request":request, 
 "news": newstring,
 "prediction" : predict[0]})
