#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install numpy
#!pip install --user -U nltk
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#!pip install unidecode
#!pip install matplotlib
#!pip install pyhive
#!pip install seaborn
#!pip install wordcloud


# In[2]:


import pandas as pd
import nltk
import time
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import RSLPStemmer
from unidecode import unidecode
from pyhive import hive
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud, STOPWORDS 
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score


warnings.filterwarnings("ignore")


# In[3]:


#Carrega dados raw do csv
df_full = pd.read_csv("mergedfile_v3.csv", names=["schema", "wiki_id", "type", "namespace", "title", "comment", "timestamp", "user", "bot", "minor", "patrolled", "server_url", "server_name", "server_script_path", "wiki", "parsedcomment"])

df = df_full[["wiki_id", "title", "timestamp", "user","bot"]]
# Remove duplicados
df = df.drop_duplicates()
df.head()


# ## Pré Processamento

# In[4]:


#Remove linhas com dados nulos, transformando em minusculo e removendo aspas simples
df.dropna(inplace=True)
df["user"] = df["user"].replace({'\'': ''}, regex=True)
lw_text = df["title"].str.lower()
lw_text = lw_text.replace({'\'': ''}, regex=True)
df["user"] = df["user"].str.strip().replace({'^[0-9]*$': 'unknown'}, regex=True)


# In[5]:


#Normalizando timestamp
s = "01/01/2020"
default_timestamp = int(datetime.datetime.strptime('01/01/2020', '%d/%m/%Y').strftime("%s"))
df["timestamp"] = df["timestamp"].str.strip().replace({'^((?![0-9]).)*$': default_timestamp}, regex=True)


# In[6]:


#Substitui valores diferentes de booleano pela item de maior frequencia
#max_freq = df.bot.mode()[0]
#df["bot"] = df["bot"].str.strip().replace({'^((?!(False|True)).)*$': max_freq.strip()}, regex=True)


# In[7]:


#Cria os tokens dos titulos
tokens =  lw_text.apply(word_tokenize)


# In[8]:


#Normalizando com unicode
tokens_uni = tokens.apply(lambda x: [unidecode(z) for z in x ])


# In[9]:


#Remove stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords.extend(['categoria','artigos', 'predefinicao','ficheiro','sobre','predefinicoes','redirecionamentos','esbocos','ligados','elemento','inexistentes','ficheiros','usuario','wikidata','paginas','wikipedia','discussao','lista'])
stopwords = set(stopwords + list(punctuation))
title_cleaned = tokens_uni.apply(lambda line:  [w for w in line if not w in stopwords])


# In[10]:


#Cria coluna com os titulos tratados
df["title_cleaned"] = title_cleaned.apply(lambda line: " ".join(line))
df.replace("", np.nan, inplace=True)
df.dropna(inplace=True)
df.head()


# In[11]:


#Cria coluna com os titulos com steamming
def Stemming(sentence):
    stemmer = RSLPStemmer()
    phrase = []
    for word in sentence:
        phrase.append(stemmer.stem(word.lower()))
    return phrase

stemmed_list = title_cleaned.apply(lambda line: Stemming(line))
df["title_stemmed"] = stemmed_list.apply(lambda line: " ".join(line))
df.head()


# ## Análises

# In[12]:


# Separa dataframe entre alterados e não alterados por bot
title_bot_true_full = df[df.bot==True]
title_bot_false_full = df[df.bot==False]


# ## Gráfico da quantidade de registros

# In[13]:


# Vec. transform com bots = true
str_list = title_bot_true_full["title_stemmed"].values
vec = TfidfVectorizer()
vec.fit(str_list)
features_bot_true = vec.transform(str_list)

# Vec. transform com bots = false
str_list = title_bot_false_full["title_stemmed"].values
vec.fit(str_list)
features_bot_false = vec.transform(str_list)

labels = 'Bot','Não Bots'
data = [features_bot_true.size, features_bot_false.size]

fig1, ax1 = plt.subplots()

ax1.pie(data, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

ax1.axis('equal')
ax1.set_title("Bot vs Não Bot")
plt.show()


# ## Dendograma top 100

# In[14]:


top_N = 100
words = df["title_cleaned"].str.cat(sep=' ').split()
rslt = pd.DataFrame(Counter(words).most_common(top_N), columns=['Word', 'Frequency']).set_index('Word')
print(rslt)


# In[15]:


str_list = rslt.index.values
vec = TfidfVectorizer()
vec.fit(str_list)
matrix_transf = vec.transform(str_list)

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

dist = 1 - cosine_similarity(matrix_transf)
linkage_matrix = ward(dist)

fig, ax = plt.subplots(figsize=(15, 20)) 
ax = dendrogram(linkage_matrix, orientation="left", labels=str_list);

plt.tick_params(    axis= 'x',          
    which='both',      
    bottom='off',     
    top='off',         
    labelbottom='off')

plt.tight_layout() 

plt.savefig('ward_clusters.png', dpi=200)


# ## Análise de todas as palavras

# In[16]:


## Word Cloud


# In[17]:


lst_words = ''
lst_words += " ".join(words)+ " "

wordcloud = WordCloud(width = 800, height = 800, 
                min_font_size = 10).generate(lst_words)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# ## Top 10 títulos alterados por bot

# In[18]:


#bot = true
top_N = 10
words = title_bot_true_full["title_cleaned"].str.cat(sep=' ').split()
rslt = pd.DataFrame(Counter(words).most_common(top_N), columns=['Word', 'Frequency']).set_index('Word')
print(rslt)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)


# ## Top 10 títulos não alterados por bots

# In[19]:


#bot = false
top_N = 10
words = title_bot_false_full["title_cleaned"].str.cat(sep=' ').split()
rslt = pd.DataFrame(Counter(words).most_common(top_N), columns=['Word', 'Frequency']).set_index('Word')
print(rslt)
rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)


# ## K-Means bot = false

# In[20]:


# Amostra das bases
title_bot_false = title_bot_false_full.sample(frac=0.10)
title_bot_true = title_bot_true_full.sample(frac=0.10)


# In[21]:


K = range(2,30)
for n_clusters in K:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(features_bot_false)
    centers = clusterer.cluster_centers_

    score = silhouette_score(features_bot_false, preds)
    print("Para n_clusters = {}, silhouette score é {})".format(n_clusters, score))


# In[22]:


Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(features_bot_false)
    Sum_of_squared_distances.append(km.inertia_)


# In[23]:


#Best result cluster = 150
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[24]:


fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1)

#Visualização gráfica 2D     # Converte as features para 2D     
pca = PCA(n_components=2, random_state= 0)
reduced_features = pca.fit_transform(features_bot_false.toarray())

# Converte os centros dos clusters para 2D     
reduced_cluster_centers = pca.transform(km.cluster_centers_)

#Plota gráfico 2D     
ax.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(features_bot_false))
ax.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='o', s=150, edgecolor='k')

#Plota números nos clusters     
for i, c in enumerate(reduced_cluster_centers):
    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

cluster=5
#Adiciona informações no gráfico     
plt.title("Análise de cluster k = %d" % cluster)
plt.xlabel('Dispersão em X')
plt.ylabel('Dispersão em Y')



#Visualização gráfica 3D 

ax = fig.add_subplot(1, 2, 2,projection="3d")

# ax = plt.axes(projection="3d") 
# Adiciona informações no gráfico     
plt.title("Análise de cluster k = %d" % cluster)
plt.xlabel('Dispersão em X')
plt.ylabel('Dispersão em Y')

#converte dados para 3D     
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(features_bot_false.toarray())

#Plota dados em 3D     
ax.scatter3D(reduced_features[:,0], reduced_features[:,1], reduced_features[:,2], marker='o', s=150, edgecolor='k', c=km.predict(features_bot_false))

# Converte os centros dos clusters para 3D     
reduced_cluster_centers = pca.transform(km.cluster_centers_)

#Salva arquivo de imagem 3D     
plt.savefig("grafico_cluster_k=%d" % cluster)
plt.show()


# ## K-means bot = True

# In[25]:


K = range(2,30)
for n_clusters in K:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(features_bot_true)
    centers = clusterer.cluster_centers_

    score = silhouette_score(features_bot_true, preds)
    print("Para n_clusters = {}, silhouette score é {})".format(n_clusters, score))


# In[26]:


Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(features_bot_true)
    Sum_of_squared_distances.append(km.inertia_)


# In[27]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[28]:


fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1)

#Visualização gráfica 2D     # Converte as features para 2D     
pca = PCA(n_components=2, random_state= 0)
reduced_features = pca.fit_transform(features_bot_true.toarray())

#Distancias calculadas com o fit_transform
reduced_features

# Converte os centros dos clusters para 2D     
reduced_cluster_centers = pca.transform(km.cluster_centers_)

#Plota gráfico 2D     
ax.scatter(reduced_features[:,0], reduced_features[:,1], c=km.predict(features_bot_true))
ax.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='o', s=150, edgecolor='k')

#Plota números nos clusters     
for i, c in enumerate(reduced_cluster_centers):
    ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

cluster=5
#Adiciona informações no gráfico     
plt.title("Análise de cluster k = %d" % cluster)
plt.xlabel('Dispersão em X')
plt.ylabel('Dispersão em Y')

#Visualização gráfica 3D 

ax = fig.add_subplot(1, 2, 2,projection="3d")

# ax = plt.axes(projection="3d") 
# Adiciona informações no gráfico     
plt.title("Análise de cluster k = %d" % cluster)
plt.xlabel('Dispersão em X')
plt.ylabel('Dispersão em Y')

#converte dados para 3D     
pca = PCA(n_components=3, random_state=0)
reduced_features = pca.fit_transform(features_bot_true.toarray())

#Plota dados em 3D     
ax.scatter3D(reduced_features[:,0], reduced_features[:,1], reduced_features[:,2], marker='o', s=150, edgecolor='k', c=km.predict(features_bot_true))

# Converte os centros dos clusters para 3D     
reduced_cluster_centers = pca.transform(km.cluster_centers_)

#Salva arquivo de imagem 3D     
plt.savefig("grafico_cluster_k=%d" % cluster)
plt.show()


# ## DBScan

# ### bot = true

# In[29]:


X = features_bot_true.toarray(); 
data = pd.DataFrame(X)
cor = data.corr()

fig = plt.figure(figsize=(10,10))
sns.heatmap(cor, square = True); plt.show()

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Conduct DBSCAN Clustering
clt = DBSCAN()

# Train model
model = clt.fit(X_std)

# Predict clusters
clusters = pd.DataFrame(model.fit_predict(X_std))
data['Cluster'] = clusters

# Visualise cluster membership
fig = plt.figure(figsize=(10,10)); 
ax = fig.add_subplot(111)
scatter = ax.scatter(data[0],data[1], c=data['Cluster'],s=50)
ax.set_title('DBSCAN Clustering - bot = true')
ax.set_xlabel('X0'); 
ax.set_ylabel('X1')
plt.colorbar(scatter); 
plt.show()


# ### bot = False

# In[ ]:


X = features_bot_false.toarray(); 
data = pd.DataFrame(X)
cor = data.corr()

fig = plt.figure(figsize=(10,10))
sns.heatmap(cor, square = True); plt.show()

# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Conduct DBSCAN Clustering
clt = DBSCAN()

# Train model
model = clt.fit(X_std)

# Predict clusters
clusters = pd.DataFrame(model.fit_predict(X_std))
data['Cluster'] = clusters

# Visualise cluster membership
fig = plt.figure(figsize=(10,10)); 
ax = fig.add_subplot(111)
scatter = ax.scatter(data[0],data[1], c=data['Cluster'],s=50)
ax.set_title('DBSCAN Clustering - bot = false')
ax.set_xlabel('X0'); 
ax.set_ylabel('X1')
plt.colorbar(scatter); 
plt.show()


# In[ ]:




