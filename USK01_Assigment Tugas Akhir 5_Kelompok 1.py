#!/usr/bin/env python
# coding: utf-8

# Nama Peserta :
# - Muhammad Mahyadi
# - Zahara Rumaisya
# - Bahrul Ulumul Haq
# 
# Universitas Host : Universitas Syiah Kuala
# 
# Kelas : USK01
# 
# Kelompok : 1
# 
# Tema Project Kelompok : Emisi Indonesia

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv("D:/coobaa.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum().sort_values(ascending=False)


# In[6]:


df.shape


# In[7]:


fig = px.histogram(df, x='Provinsi', y='Populasi')
fig.update_layout(title_text='Jumlah Populasi Penduduk berdasarkan Provinsi')
fig.show()


# In[8]:


fig = px.histogram(df, x='Provinsi', y='Epk')
fig.update_layout(title_text='Emisi Per Kapita berdasarkan Provinsi')
fig.show()


# In[9]:


fig = px.histogram(df, x='Provinsi', y='Intensitas_em')
fig.update_layout(title_text='Intensitas Emisi berdasarkan Provinsi')
fig.show()


# In[10]:


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] -             bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
       
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels):
        
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')


bubble_chart = BubbleChart(area=df['ETDT'],
                           bubble_spacing=0.1)

bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=(15, 15))
bubble_chart.plot(
    ax, df['Provinsi'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('ETDT')

plt.show()


# In[11]:


bubble_chart = BubbleChart(area=df['ETDETI'],
                           bubble_spacing=0.1)

bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=(15, 15))
bubble_chart.plot(
    ax, df['Provinsi'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('ETDETI')

plt.show()


# In[12]:


bubble_chart = BubbleChart(area=df['ETDL'],
                           bubble_spacing=0.1)

bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"), figsize=(15, 15))
bubble_chart.plot(
    ax, df['Provinsi'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('ETDL')

plt.show()


# In[13]:


plt.figure(figsize=(10,7))
labels=['No','Yes']
colors = ["Maroon","khaki"]
plt.pie(df['Aksi_Adaptasi'].value_counts(),labels=labels,colors=colors,
        autopct='%1.2f%%', shadow=True, startangle=140) 
plt.title('Aksi Adaptasi', loc='center', fontsize=15)
plt.show()


# In[14]:


a = np.random.rand(34)
plt.figure(figsize=(20,8))
plt.scatter(df['Epk'], df['Intensitas_em'], s=a*2000, alpha=0.4)
plt.xlabel("Emisi per kapita")
plt.ylabel("Intensitas Emisi")
plt.title("EPK vs IE")
plt.show()


# ## Pembentukan Model

# In[15]:


#Library membuat model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.model_selection import train_test_split


# In[16]:


pip install "missingno"


# In[17]:


import missingno as msno
msno.bar(df,figsize=(8,6),color='skyblue')
plt.show()


# Dataset INDONESIA_COVID19.csv ini adalah dataset yang telah melewati proses cleansing sehingga dataset ini sudah bersih

# In[18]:


X=df.iloc[:,2:8].values
y=df.iloc[:,1].values


# transform label data dengan menggunakan library LabelEncoder

# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Split data train dan test dengan function train_test_split() dengan test_size=0.3 dan random_state=0

# In[20]:


#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Training model menggunakan training data yang sudah displit sebelumnya.

# In[21]:


#Membuat model
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(df[['Luas_hutan']])
train_y = np.asanyarray(df[['Intensitas_em']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# Dari nilai diatas, kalau dimasukan ke dalam rumus persamaan menjadi:
# #     **y = 0.34x + 39.35**

# #INTERPRETASI

# Visualisasi Regression Line

# In[22]:


plt.scatter(df['Luas_hutan'], df['Intensitas_em'], color='olive')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], color='red')
plt.xlabel("Luas_hutan")
plt.ylabel("Intensitas_em")
plt.show()


# Garis merah merupakan Regression Line dari model yang telah dibuat sebelumnya.

# In[23]:


#library
import pandas as pd
import numpy as np
import joblib 
from sklearn.linear_model import LinearRegression


# In[24]:


#dataset 
df = pd.read_csv("D:/coobaa.csv")


# In[25]:


#Library membuat model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.model_selection import train_test_split


# In[26]:


X=df.iloc[:,2:8].values
y=df.iloc[:,1].values


# In[27]:


#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[28]:


#call model regression
X = np.asanyarray(df[['Luas_hutan']])
Y = np.asanyarray(df[['Intensitas_em']])
model = LinearRegression().fit(X,Y)
model


# In[37]:


#save model
filename = 'model.h5'
joblib.dump(model, filename)


# In[38]:


#load model
loaded_model = joblib.load(filename)


# In[39]:


#prediction model
loaded_model.predict(np.array([10]).reshape(1, 1))


# In[33]:


import pickle


# In[40]:


#save
pickle_out = open("emisi_flask.pkl", "wb")
pickle.dump(model, pickle_out)
loaded_model = pickle.load(open("emisi_flask.pkl", "rb"))
result = loaded_model.score(X , Y)
print(result)

