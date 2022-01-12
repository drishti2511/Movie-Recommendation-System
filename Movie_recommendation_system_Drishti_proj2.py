#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Movie Recommender System
#Project2_Drishti


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


movies_df=pd.read_csv("/Users/sushmajain/Desktop/Movie_recom_system/movies.csv")
credits_df=pd.read_csv("/Users/sushmajain/Desktop/Movie_recom_system/credits.csv")


# In[4]:


movies_df.head()


# In[5]:


credits_df.head()


# In[6]:


movies_df.shape


# In[7]:


credits_df.shape


# In[8]:


df=movies_df.merge(credits_df,on='title') #merging movies and credits data set on the basis of titles
df.shape #since joined on the basis of title hence total columns=20+4-1


# In[9]:


#remove column which are not important for analysis
df.head()


# In[10]:


#high budget can not be a criteria
#genere is ismportant
#id would be used to make website
#keywords - highly important
#original lang not required as shown below
#original title is not useful as it will have names in regional language
#title is useful
#overview is important to cateogarize movies as same
#though popularity is an important measure but as we are finding similarities on content basis hence it wont be considered
#production companies not important
#places where shooting occurs
#release dates are important (older poplutions would like 90s movies)...though not cosidering it
#revenue is numerical...wont fit here
#runtime does not matter
#status dont matter
#spoken lang dont matter
#tagline is vague hence not kept
#vote average and vote count are ignored
#cast and crew are important 

#columns to be counted
#id,title,genere,cast,crew,overview,keywords


# In[11]:


df['original_language'].value_counts() #as count of english movies is very high hence no need to keep original lang column


# In[12]:


df=df[['movie_id','title','overview','genres','keywords','cast','crew']] #useful columns


# In[13]:


df.head() #ab ispe hi kaam kareinge


# In[14]:


#now we will make only three columns :-movie_id,title and tags(consisting of last 5 columns
#after processing their datas)


# In[15]:


df.isnull().sum()


# In[16]:


df.dropna(inplace=True) #removing movies whose overview wasn't available
df.isnull().sum()


# In[17]:


df.duplicated().sum() #checking for duplicate rows


# In[18]:


df.iloc[1].genres


# In[19]:


#we have to convert this list of dictionary to a list
#it is a list of strings ...we need to convert it to integer list using literal_eval from ast library


# In[20]:


import ast


# In[21]:


def convertor(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l #having indentation here will change the result


# In[22]:


df['genres']=df['genres'].apply(convertor)


# In[23]:


df.head()


# In[24]:


#we will do the same thing on keywords column
df['keywords']=df['keywords'].apply(convertor)


# In[25]:


#for cast and crew we will be taking top 3 persons(first three dictionaries from each list)
def convertor_req(obj):
    l=[]
    x=0  
    for i in ast.literal_eval(obj):
        l.append(i['name'])
        x=x+1
        if(x==3):
            break
    return l
convertor_req(df.iloc[1].cast)


# In[26]:


df['cast']=df['cast'].apply(convertor_req)
#df['crew']=df['crew'].apply(convertor_req)
df.head()


# In[27]:


def director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
#only director's name is important in the crew


# In[28]:


df['crew']=df['crew'].apply(director)


# In[29]:


df.head() 


# In[30]:


#converting overview to list
df['overview']=df['overview'].apply(lambda x:x.split())


# In[31]:


df.head() 


# In[32]:


#Now we have to remove white spaces from a name...i.e convert sam mendes to sammendes otherwise model will recommend
#sam worgingthton also alongwidth sam mendes....this is a transfromation we have to apply
df['genres']=df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
df['cast']=df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
df['crew']=df['crew'].apply(lambda x:[i.replace(" ","") for i in x])
df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
df.head()


# In[33]:


df['tags']=df['overview']+df['genres']+df['keywords']+df['cast']+df['crew']


# In[34]:


df=df.drop(['overview','genres','keywords','cast','crew'],axis=1)


# In[35]:


df['tags']=df['tags'].apply(lambda x:" ".join(x)) #joining all lists and converting them to a single string


# In[36]:


df.head()


# In[37]:


df['tags']=df['tags'].apply(lambda x:x.lower())  #converting whole text to lower case


# In[38]:


df.head()


# In[39]:


import nltk
from nltk.stem.porter import PorterStemmer


# In[40]:


ps=PorterStemmer()
def stem(text):
    y=[] #to convert string to list
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y) #list to string back


# In[41]:


#stemming to reduce redundancy


# In[42]:


df['tags']=df['tags'].apply(stem)


# In[43]:


#now we have to do text vectorization (similar to what done in fake news proj)
#each movie will be converted to a vector in a n-d space and 5 nearest vectors will be our recommendations
#based on the given movies.
#we will use bag of words technique to vectorize tag column of each movie
#in fake news predictor we used tfidf...(advanced technique)

#bag of words technique:
#Firstly concatinate all tags->sab tags ko add karke ek bahut bada string bana diya
#Calculate frequency of each word
#extract top 5000 words with maximum frequency and name them as bag of words(bow)
#then we will take each tag and form an array calculating number of occurences of words for bow from that tag
#this array will be a vector in 5000X5000 dimensional space
#and now we will chose the closest 5 vectors in it for movie recommendation

#stop words won't be considered during vectorization


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english') #taking 5000 most frequent words


# In[45]:


vect=cv.fit_transform(df['tags']).toarray()


# In[46]:


cv.get_feature_names()


# In[47]:


#to check the similarity we will calculate the cosine distance i.e lesser the angle between two vectors, more
#similar they are
#we won't be using eucliadian distance(distance btw tips of the vectors)


# In[48]:


from sklearn.metrics.pairwise import cosine_similarity
sim=cosine_similarity(vect) #similarity matrix


# In[49]:


sim.shape
#we will use enumerate function to save index after sorting(basically maps)


# In[50]:


def recommend(movie):
    index=df[df['title']==movie].index[0]
    mv_list=sorted(list(enumerate(sim[index])),reverse=True,key=lambda x:x[1])[1:6] #taking first 5 movies sorted on the basis of index
    
    for i in mv_list :
        print(df.iloc[i[0]].title)
    


# In[55]:


recommend('The Notebook')


# In[56]:


#hence we have made our recommender system
#now we will make it into a website


# In[75]:


import pickle
pickle.dump(df,open('movie_list.pkl','wb'))
pickle.dump(sim,open('similarity.pkl','wb'))

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




