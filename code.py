# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:17:30 2020

@author: Parth
"""

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
# Importing the dataset
dataset = pd.read_csv(r'C:\Users\EXTCA\Desktop\BAAP-Nayasa.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 20].values
h=0.2 #step_size

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #Minkowski is the distance metric to use for tree, and with p=2 is equivalent to standard Euclidean metric
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
               


#----------------------------------

import librosa
import scipy.io.wavfile as wav
import numpy as np
import os
(sig,rate) = librosa.load(r"C:\Users\parth\Desktop\Lesser Whitethroat-4-[AudioTrimmer.com].wav")

rate=44100        # get mfcc
mfcc_feat = librosa.feature.mfcc(sig,rate)
mf1=np.transpose(mfcc_feat)
mf3=sc.transform(mf1)
#mf2=np.mean(mf1)
y_pred1 = list(classifier.predict(mf3))
print(y_pred1[0])

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 
  
 
print(most_frequent(y_pred1)) 
xyz=most_frequent(y_pred1)

from PIL import Image
if(xyz==0):
    print("This Bird is Lesser Whitethroat")
    img  = Image.open(r"C:\Users\parth\Desktop\Lesser Whitethroat.jpg")  
    img.show()
elif(xyz==1):
    print("This Bird is Golden Vireo") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Golden Vireo.jpg")  
    img.show()
elif(xyz==2):
    print("This Bird is Crescent-chested Warbler") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Crescent-chested Warbler.jpg")  
    img.show()
elif(xyz==3):
    print("This Bird is Scarlet Minivet") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Scarlet Minivet.jpg")  
    img.show()
elif(xyz==4):
    print("This Bird is Slender-billed Crow") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Slender-billed Crow.jpg")  
    img.show()
elif(xyz==5):
    print("This Bird is warbling vireo") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\warbling vireo.jpg")  
    img.show()
elif(xyz==6):
    print("This Bird is Willow Warbler") 
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Willow Warbler.jpg")  
    img.show()
elif(xyz==7):
    print("This Bird is Blyth's Paradise")
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Blyth's Paradise.jpg")  
    img.show()    
elif(xyz==8):
    print("This Bird is Common Sandpiper")
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Common Sandpiper.jpg")  
    img.show()      
elif(xyz==9):
    print("This Bird is Simeulue Scops Owl")
    img  = Image.open(r"C:\Users\EXTCA\Desktop\Birds pictures\Simeulue Scops Owl.jpg")  
    img.show()    
    
    
    
