#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[17]:


df = pd.read_csv("movie_dataset.csv")
df.head()


# In[18]:


features = ['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(original_title):
    return df[df.original_title == original_title]["index"].values[0]


# In[19]:


for feature in features:
    df[feature] = df[feature].fillna('') 

df["combined_features"] = df.apply(combine_features,axis=1)


# In[20]:


df.iloc[0].combined_features


# ### Content  Based  Filtering

# In[21]:


cv = CountVectorizer() 
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)
df.rename(columns={'Unnamed: 0': 'index'}, inplace= True)


# In[22]:


movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[23]:


sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


# In[24]:


i=0
print("Top 10 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>10:
        break


# In[ ]:




