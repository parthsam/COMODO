# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:56:15 2022

@author: parth
"""
import numpy as np
#from morphSimilarity_png import compute_distance
import numpy as np
import matplotlib.pyplot as plt
import math 
import pylab
import pandas as pd
import random
from scipy.stats import linregress
from sklearn.manifold import MDS
import sklearn
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 500

from cycler import cycler
COLORS =['#fecc5c','#bd0026','#2c7fb8','#253494','#4d9221','#c51b7d']#['#ffffcc','#a1dab4','#41b6c4','#225ea8']
default_cycler = cycler(color=COLORS)
plt.rc('axes', prop_cycle=default_cycler) 
from sklearn.cluster import DBSCAN
from scipy.integrate import simps
from numpy import trapz

from sklearn import preprocessing
from morphSimilarity import compute_distance



VS5Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\5replicas\SelectedFiles\ImageFiles\images" ))
VS10Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\10replicas\SelectedFiles\ImageFiles\images" ))
VS20Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\20replicas\SelectedFiles\ImageFiles\images" ))
VS100Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\100replicas\collectedData\ImageFiles\images" ))

VS5ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\5replicas\SelectedFiles\LayeredImageFiles\images" ))
VS10ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\10replicas\SelectedFiles\LayeredImageFiles\images" ))
VS20ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\20replicas\SelectedFiles\LayeredImageFiles\images" ))
VS100ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\100replicas\collectedData\LayeredImageFiles\images" ))


AR5Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\5replicas\SelectedFiles2\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR10Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\10replicas\SelectedFiles2\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR20Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\20replicas\SelectedFiles2\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR100Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\100replicas\collectedData\ImageFiles\images", signature_function='shape_ratio_sig' ))

AR5ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\5replicas\SelectedFiles2\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR10ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\10replicas\SelectedFiles2\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR20ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\20replicas\SelectedFiles2\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR100ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\ThinFilms\100replicas\collectedData\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))






#########400x100############

VS5Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\SelectedFiles\ImageFiles\images" ))
VS10Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\SelectedFiles\ImageFiles\images" ))
VS20Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\SelectedFiles\ImageFiles\images" ))
VS100Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\SelectedFiles\ImageFiles\images" ))

VS5ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\SelectedFiles\LayeredImageFiles\images" ))
VS10ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\SelectedFiles\LayeredImageFiles\images" ))
VS20ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\SelectedFiles\LayeredImageFiles\images" ))
VS100ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\SelectedFiles\LayeredImageFiles\images" ))


AR5Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\SelectedFiles\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR10Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\SelectedFiles\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR20Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\SelectedFiles\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR100Replicas2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\SelectedFiles\ImageFiles\images", signature_function='shape_ratio_sig' ))

AR5ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\SelectedFiles\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR10ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\SelectedFiles\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR20ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\SelectedFiles\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))
AR100ReplicasLayered2=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\SelectedFiles\LayeredImageFiles\images", signature_function='shape_ratio_sig'))

VS20Replicas2ct=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\ChangedThreshold\ImageFiles\images" ))
VS20ReplicasLayered2ct=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\ChangedThreshold\LayeredImageFiles\images" ))
AR20Replicas2ct=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\ChangedThreshold\ImageFiles\images", signature_function='shape_ratio_sig' ))
AR100ReplicasLayered2ct=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\ChangedThreshold\LayeredImageFiles\images", signature_function='shape_ratio_sig' ))

#######TPCC################


from two_point_correlation import compute_distance

TPC5replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\SelectedFiles\ImageFiles\images" ))
TPC10replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\SelectedFiles\ImageFiles\images" ))
TPC20replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\SelectedFiles\ImageFiles\images" ))
TPC100replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\SelectedFiles\ImageFiles\images" ))


TPC5replicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\5replicas\collectedData\LayeredImageFiles\images" ))
TPC10replicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\10replicas\collectedData\LayeredImageFiles\images" ))
TPC20replicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\20replicas\collectedData\LayeredImageFiles\images" ))
TPC100replicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas\collectedData\LayeredImageFiles\images" ))






from morphSimilarity import compute_distance
vs38=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\3.8",visualize_graphs=True))
AR38=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\3.8", signature_function='shape_ratio_sig'))

vs22=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\2.2",visualize_graphs=True))
AR22=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\2.2", signature_function='shape_ratio_sig'))

from two_point_correlation import compute_distance
TPC38=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\3.8"))
TPC22=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\2.2"))


###########################MDS######################################



# for i in range(0,100):
#     plt.scatter(X_transformed[i,0], X_transformed[i,1],c =label_color[i],marker=('+'))
# for i in range(100,200):

from scipy.io import savemat

Distance=vs38

fig = plt.figure()
fig.patch.set_facecolor('white')
embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(Distance[:])


# plt.scatter(X_transformed[0:5,0],X_transformed[0:5,1] )
# plt.scatter(X_transformed[5:10,0],X_transformed[5:10,1] )
# plt.scatter(X_transformed[10:15,0],X_transformed[10:15,1] )
# plt.scatter(X_transformed[15:20,0],X_transformed[15:20,1] )
# plt.scatter(X_transformed[20:25,0],X_transformed[20:25,1] )
# plt.scatter(X_transformed[25:30,0],X_transformed[25:30,1] )


# plt.scatter(X_transformed[0:10,0],X_transformed[0:10,1] )
# plt.scatter(X_transformed[10:20,0],X_transformed[10:20,1] )
# plt.scatter(X_transformed[20:30,0],X_transformed[20:30,1] )
# plt.scatter(X_transformed[30:40,0],X_transformed[30:40,1] )
# plt.scatter(X_transformed[40:50,0],X_transformed[40:50,1] )
# plt.scatter(X_transformed[50:60,0],X_transformed[50:60,1] )



# plt.scatter(X_transformed[0:20,0],X_transformed[0:20,1] )
# plt.scatter(X_transformed[20:40,0],X_transformed[20:40,1] )
# plt.scatter(X_transformed[40:60,0],X_transformed[40:60,1] )
# plt.scatter(X_transformed[60:80,0],X_transformed[60:80,1] )
# plt.scatter(X_transformed[80:100,0],X_transformed[80:100,1] )
# plt.scatter(X_transformed[100:120,0],X_transformed[100:120,1] )


plt.scatter(X_transformed[0:100,0],X_transformed[0:100,1] )
plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
plt.scatter(X_transformed[200:300,0],X_transformed[200:300,1] )
# plt.scatter(X_transformed[300:400,0],X_transformed[300:400,1] )
# plt.scatter(X_transformed[400:500,0],X_transformed[400:500,1] )
# plt.scatter(X_transformed[500:600,0],X_transformed[500:600,1] )




# plt.scatter(X_transformed[0:100,0],X_transformed[0:100,1] )
# plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
# plt.scatter(X_transformed[200:300,0],X_transformed[200:300,1] )

# plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
# for i in range(0,100):
#     print(i)
#     plt.text(X_transformed[100+i,0], X_transformed[100+i,1], str(i))



# for i in range(X_transformed.shape[0]):
#     plt.text(X_transformed[i,0], X_transformed[i,1], str(i))

plt.rc('font', size=12) 
plt.legend(['Phi0.51Chi3.8','Phi0.54Chi3.8','Phi0.60Chi3.8'])
#plt.legend(['Phi0.51Chi2.2','Phi0.51Chi3.8','Phi0.54Chi2.2', 'Phi0.54Chi3.8','Phi0.60Chi2.2', 'Phi0.60Chi3.8','Phi0.60Chi3.8']);
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS analysis ')



#     plt.scatter(X_transformed[i,0], X_transformed[i,1],c =label_color[i],marker=('s'))
# for i in range(200,300):
#     plt.scatter(X_transformed[i,0], X_transformed[i,1],c =label_color[i],marker=('o'))
# for i in range(300,400):
#     plt.scatter(X_transformed[i,0], X_transformed[i,1],c= label_color[i],marker=('^'))



from morphSimilarity import compute_distance
Test=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles\images\3.8\Test"))


plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
plt.scatter(X_transformed[1:5,0],X_transformed[1:5,1] )
plt.scatter(X_transformed[5:6,0],X_transformed[5:6,1] )
plt.scatter(X_transformed[6:7,0],X_transformed[6:7,1] )
plt.scatter(X_transformed[7:8,0],X_transformed[7:8,1] )
plt.scatter(X_transformed[8:9,0],X_transformed[8:9,1] )


plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
for i in range(100,200):
    plt.text(X_transformed[i,0], X_transformed[i,1], str(i))

plt.rc('font', size=12) 
#plt.legend(['Phi0.50Chi2.2','Phi0.50Chi3.8', 'Phi0.54Chi2.2','Phi0.54Chi3.8', 'Phi0.60Chi2.2','Phi0.60Chi3.8']);
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.title('MDS analysis ')
