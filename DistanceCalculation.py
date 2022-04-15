# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:44:11 2021

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
min_max_scaler = preprocessing.MinMaxScaler()
from morphSimilarity import compute_distance

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

#from morphSimilarity_png import compute_distance


DistanceTFSurfacetoVolume = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ThinFilms"))
DistanceMASSurfacetoVolume = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ModerateAspectRatio"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASSurfacetoVolume = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))


DistanceTFAspectRatio = np.array(compute_distance(r'C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ThinFilms', signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASAspectRatio = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ModerateAspectRatio", signature_function='shape_ratio_sig' , visualize_graphs=True))
DistanceEASAspectRatio = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio", signature_function='shape_ratio_sig' , visualize_graphs=False))



DistanceTFSurfacetoVolumeLayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ThinFilms\LayeredImages\images"))
DistanceTFAspectRatioLayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ThinFilms\LayeredImages\images", signature_function='shape_ratio_sig' , visualize_graphs=False))





################20 replicas###########
DistanceTFSurfacetoVolumeExpanded = np.array(compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata'))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASSurfacetoVolumeExpanded = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\MASexpandeddata"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASSurfacetoVolumeExpanded = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASexpandeddata"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))



DistanceTFAspectRatioExpanded = np.array(compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata', signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASAspectRatioExpanded = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\\Expanded Data\MASexpandeddata", signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceEASAspectRatioExpanded = np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASexpandeddata", signature_function='shape_ratio_sig' , visualize_graphs=False))

DistanceTFSurfacetoVolumeExpandedLayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata\LayeredImages\images"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceTFAspectRatioExpandedLayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata\LayeredImages\images", signature_function='shape_ratio_sig' , visualize_graphs=True))




############100replicas(400 total)######################

# DistanceTFstv100replicas = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\ThinFlims",cosine=True))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
# DistanceMASstv100replicas = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\MAS",cosine=True))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
# DistanceEASstv100replicas = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\EAS",cosine=True,visualize_graphs=True))#, signature_function='shape_ratio_sig' , visualize_graphs=False))


DistanceTFstv100replicasE = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\ThinFlims"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASstv100replicasE = np.load('DistanceMASstv100replicasE.npy')
DistanceEASstv100replicasE =  np.load('DistanceEASstv100replicasE.npy')




DistanceTFAR100replicasE = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\ThinFlims", signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASAR100replicasE = np.load("DistanceMASAR100replicasE.npy")
DistanceEASAR100replicasE = np.load("DistanceEASAR100replicasE.npy")


DistanceTFstv100replicasELayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\ThinFlims\LayeredImages\images"))#, signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceTFAR100replicasELayered = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\100 replicas\ThinFlims\LayeredImages\images", signature_function='shape_ratio_sig' , visualize_graphs=False))

############200replicas(800 total)######################



#DistanceTFstv200replicasE = np.load('DistanceTFstv200replicasE.npy')
DistanceMASstv200replicasE =np.load('DistanceMASstv200replicasE.npy')
DistanceEASstv200replicasE =np.load('DistanceEASstv200replicasE.npy')




#DistanceTFAR200replicasE = np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\200replicas\ThinFlims", signature_function='shape_ratio_sig' , visualize_graphs=False))
DistanceMASAR200replicasE = np.load('DistanceMASAR200replicasE.npy')
DistanceEASAR200replicasE = np.load('DistanceEASAR200replicasE.npy')







from two_point_correlation import compute_distance
TPCDistanceTF= compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ThinFilms')
TPCDistanceMAS= compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ModerateAspectRatio')
TPCDistanceEAS= compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio')
TPCDistanceTFexpanded = compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\TFexpandeddata')
TPCDistanceMASexpanded = compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\MASexpandeddata")
TPCDistanceEASexpanded = compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata")

#TPCDistanceTF100replicas = np.load('TPCTF100replicas.npy')
TPCDistanceMAS100replicas = np.load('TPCMAS100replicas.npy')
TPCDistanceEAS100replicas = np.load('TPCEAS100replicas.npy')

#TPCDistanceTF200replicas = np.load('TPCTF200replicas.npy')
TPCDistanceMAS200replicas = np.load('TPCMAS200replicas.npy')
TPCDistanceEAS200replicas = np.load('TPCEAS200replicas.npy')


TPCDistance200replicastf=compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\200replicas')
SDTF=compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\SD400x100\100replicas2\SelectedFiles\ImageFiles\images')



DVS=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Initial",cosine=True, visualize_graphs=True))
DAS=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Initial",cosine=True,signature_function='shape_ratio_sig'))
DFR=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Initial",cosine=True,signature_function='fractal_dimension', visualize_graphs=True))
 


DVSF=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Final",cosine=True, visualize_graphs=True))
DASF=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Final",cosine=True,signature_function='shape_ratio_sig', visualize_graphs=True))
DFRF=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\check\Fractal\Final",cosine=True,signature_function='fractal_dimension_sig', visualize_graphs=True))
 

Dfractalvs=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\OmegaFiles\Test\Fractalcheck", visualize_graphs=True))
DfractalAR=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\OmegaFiles\Test\Fractalcheck",signature_function='shape_ratio_sig', visualize_graphs=True))
DfractalF=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\OmegaFiles\Test\Fractalmixed",signature_function='fractal_dimension_sig', visualize_graphs=True))

##################SDFRACRALIZED DATA


DfractalvsSD=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata"))
DfractalARSD=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata",signature_function='shape_ratio_minbymax_sig'))
DfractalFSD=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata",signature_function='fractal_dimension_sig', visualize_graphs=True))

DfractalvsSDC=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata"))
DfractalARSDC=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata",signature_function='shape_ratio_minbymax_sig'))
DfractalFSDC=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Dendrites\SD\SDThinFilms\Fractalizeddata",signature_function='fractal_dimension_sig'))



###############NewVolumeFractionData#############

Dstv4Replicas=np.array(compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\4Replicas'))
Dstv20Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\20replicas"))
Dstv100Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\100replicas"))
Dstv200Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\200replicas"))

DAR4Replicas=np.array(compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\4Replicas',signature_function='shape_ratio_sig', visualize_graphs=True))
DAR20Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\20replicas",signature_function='shape_ratio_sig', visualize_graphs=True))
DAR100Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\100replicas",signature_function='shape_ratio_sig', visualize_graphs=True))
DAR200Replicas=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\200replicas",signature_function='shape_ratio_sig', visualize_graphs=True))

Dstv100ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\100replicas\LayeredImages\images")) 
DAR100ReplicasLayered=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\100replicas\LayeredImages\images",signature_function='shape_ratio_sig', visualize_graphs=True))


Dstv200ReplicasLayered=np.load("TF200replicas1200VS.npy") 
DAR200ReplicasLayered=np.load("TF200replicas1200AR.npy")




###########################MDS######################################


from scipy.io import savemat

#from mpl_toolkits import mplot3d



Distance=SDTF


fig = plt.figure()
fig.patch.set_facecolor('white')
embedding = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(Distance[:])
#mymat={'X_transformed':X_transformed}
#savemat("STVEAS100replicas.mat",mymat)

#ax = plt.axes(projection ="3d")

# plt.scatter(X_transformed[0:20,0],X_transformed[0:20,1])
# plt.scatter(X_transformed[20:40,0],X_transformed[20:40,1] )
# plt.scatter(X_transformed[40:60,0],X_transformed[40:60,1] )
# plt.scatter(X_transformed[60:80,0],X_transformed[60:80,1] )
# ##

# plt.scatter(X_transformed[0:100,0],X_transformed[0:100,1] )
# plt.scatter(X_transformed[100:200,0],X_transformed[100:200,1] )
# plt.scatter(X_transformed[200:300,0],X_transformed[200:300,1] )
# plt.scatter(X_transformed[300:400,0],X_transformed[300:400,1] )

# plt.scatter(X_transformed[0:200,0],X_transformed[0:200,1] )
# plt.scatter(X_transformed[200:400,0],X_transformed[200:400,1] )
# plt.scatter(X_transformed[400:600,0],X_transformed[400:600,1] )
# plt.scatter(X_transformed[600:800,0],X_transformed[600:800,1] )
# # plt.scatter(X_transformed[800:1000,0],X_transformed[800:1000,1] )
# # plt.scatter(X_transformed[1000:1200,0],X_transformed[1000:1200,1] )

# plt.scatter(X_transformed[0:23,0],X_transformed[0:23,1] )
# plt.scatter(X_transformed[23:46,0],X_transformed[23:46,1] )
# ###plt.scatter(X_transformed[18,0],X_transformed[18,1] )



# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[0:4,0],X_transformed[0:4,1] )
# plt.scatter(X_transformed[4:8,0],X_transformed[4:8,1] )
# plt.scatter(X_transformed[8:12,0],X_transformed[8:12,1] )
# plt.scatter(X_transformed[12:16,0],X_transformed[12:16,1] )
# plt.scatter(X_transformed[12:16,0],X_transformed[12:16,1] )
# plt.scatter(X_transformed[12:16,0],X_transformed[12:16,1] )





# for i in range(X_transformed.shape[0]):
#     plt.text(X_transformed[i,0], X_transformed[i,1], str(i))
plt.rc('font', size=12) 
#plt.legend(['GrainSize(10,10)', 'GrainSize(10,80)', 'GrainSize(80,10)','GrainSize(80,80)']);
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
#['GrainSize(100,100),VF0.7,0.3', 'GrainSize(100,100),VF0.55,0.45','GrainSize(100,40),VF0.55,0.45', 'GrainSize(100,40),VF0.7,0.3', 'GrainSize(40,40),VF0.55,0.45','GrainSize(40,40),VF0.7,0.3']
#[ 'Intial Data', 'Fractalized Data'])#,'Outgroup'
#[ 'GrainSize(10,80)', 'GrainSize(20,60)','GrainSize(40,20),'GrainSize(50,50)']
#['GrainSize(20,40)', 'GrainSize(40,20)', 'GrainSize(40,40)','GrainSize(80,80)']
#['GrainSize(10,10)', 'GrainSize(10,80)', 'GrainSize(80,10)','GrainSize(80,80)']
plt.title('MDS projection  ')



######################UPGMA##########################################


######################NORMALIZED MATRIX######################

NorTFVS =  NormalizeData(DistanceTFSurfacetoVolume)
NorMASVS=  NormalizeData(DistanceMASSurfacetoVolume)
NorEASVS=  NormalizeData(DistanceEASSurfacetoVolume)

ASNorTF=  NormalizeData(DistanceTFAspectRatio)
ASNorMAS=  NormalizeData(DistanceMASAspectRatio)
ASNorEAS=  NormalizeData(DistanceEASAspectRatio)




TPCnorTF=  NormalizeData(TPCDistanceTF)
TPCnorMAS=  NormalizeData(TPCDistanceMAS)
TPCnorEAS=  NormalizeData(TPCDistanceEAS)


NorTFVSExpanded =  NormalizeData(DistanceTFSurfacetoVolumeExpanded)
NorMASVSExpanded=  NormalizeData(DistanceMASSurfacetoVolumeExpanded)
NorEASVSExpanded=  NormalizeData(DistanceEASSurfacetoVolumeExpanded)

ASNorTFExpanded=  NormalizeData(DistanceTFAspectRatioExpanded)
ASNorMASExpanded=  NormalizeData(DistanceMASAspectRatioExpanded)
ASNorEASExpanded=  NormalizeData(DistanceEASAspectRatioExpanded)




TPCnorTFExpanded=  NormalizeData(TPCDistanceTFexpanded)
TPCnorMASExpanded=  NormalizeData(TPCDistanceMASexpanded)
TPCnorEASExpanded=  NormalizeData(TPCDistanceEASexpanded)



#NorTFVS100replicas =  NormalizeData(DistanceTFstv100replicasE)
NorMASVS100replicas= NormalizeData(DistanceMASstv100replicasE)
NorEASVS100replicas=  NormalizeData(DistanceEASstv100replicasE)

#ASNorTF100replicas=  NormalizeData(DistanceTFAR100replicasE)
ASNorMAS100replicas=  NormalizeData(DistanceMASAR100replicasE)
ASNorEAS100replicas=  NormalizeData(DistanceEASAR100replicasE)




#TPCnorTF100replicas=  NormalizeData(TPCDistanceTF100replicas)
TPCnorMAS100replicas=  NormalizeData(TPCDistanceMAS100replicas)
TPCnorEAS100replicas=  NormalizeData(TPCDistanceEAS100replicas)



#NorTFVS200replicas =  NormalizeData(DistanceTFstv200replicasE)
NorMASVS200replicas=  NormalizeData(DistanceMASstv200replicasE)
NorEASVS200replicas=  NormalizeData(DistanceEASstv200replicasE)

#ASNorTF200replicas=  NormalizeData(DistanceTFAR200replicasE)
ASNorMAS200replicas=  NormalizeData(DistanceMASAR200replicasE)
ASNorEAS200replicas=  NormalizeData(DistanceEASAR200replicasE)




#TPCnorTF200replicas=  NormalizeData(TPCDistanceTF200replicas)
TPCnorMAS200replicas=  NormalizeData(TPCDistanceMAS200replicas)
TPCnorEAS200replicas=  NormalizeData(TPCDistanceEAS200replicas)



NorTF200replicasVS=NormalizeData(Dstv200ReplicasLayered)
NorTF200replicasAR=NormalizeData(DAR200ReplicasLayered)
NorTF200replicasTPC=NormalizeData(TPCDistance200replicastf)

SVS=NormalizeData(vs)
SAR=NormalizeData(AR)
STPC=NormalizeData(TPC)



#############RAND INDEX###########


Truelabelsstv=[0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2]


tfmasexplabels=np.loadtxt('TFMASLABELSEXPANDED.txt')
easexpandedlabels=np.loadtxt('EASLABELSEXPANDED.txt')

tfmasexplabels100replicas=np.loadtxt('TFMASLABELS100replicas.txt')
clusters4100replicas=np.loadtxt('100replicas4cluster.txt')
easexpandedlabels100replicas=np.loadtxt('EASLABELS100replicas.txt')

tfmasexplabels200replicas=np.loadtxt('TFMASLABELS200replicas.txt')
tfmasexplabels200replicas4clusters=np.loadtxt('MAS200replicas4clusterslabels.txt')

easexpandedlabels200replicas=np.loadtxt('EASLABELS200replicas.txt')

tfnewdata200replicaslables=np.loadtxt('TFnewdata200replicas.txt')
SDL=np.loadtxt('SDlables.txt')



gofmatTFVS=np.array([])
gofmatTFAR=np.array([])
gofmatTFTPC=np.array([])
#gof=np.zeros([100])
j=0
DistanceVS=vs
DistanceAR=AR
DistanceTPC=TPC
#print(cod)
a=np.linspace(0.1,1,100)
for i in range (0,100):
    dbVS=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceVS)
    dbAR=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceAR)
    dbTPC=DBSCAN(eps=a[i], metric='precomputed').fit(DistanceTPC)
    labelsVS = dbVS.labels_
    labelsAR = dbAR.labels_
    labelsTPC = dbTPC.labels_
    #print(labelsVS)
    gofVS=sklearn.metrics.adjusted_rand_score(SDL,labelsVS)
    gofAR=sklearn.metrics.adjusted_rand_score(SDL,labelsAR)
    gofTPC=sklearn.metrics.adjusted_rand_score(SDL,labelsTPC)
    
   # print(gof)
    gofmatTFVS=np.append(gofmatTFVS,gofVS)
    gofmatTFAR=np.append(gofmatTFAR,gofAR)
    gofmatTFTPC=np.append(gofmatTFTPC,gofTPC)
    
    #print(gof)
plt.plot(a,gofmatTFVS)
plt.plot(a,gofmatTFAR)
plt.plot(a,gofmatTFTPC)    
#plt.scatter(a,gofmat)
plt.xlabel('Distance')
plt.ylabel('Rand Index')
plt.title('Distance v/s Rand Index')
plt.legend(['Surface/Volume','Apect Ratio','Two Point coorelation'])
    


areaSV = trapz(gofmatTFVS, a)
areaAR = trapz(gofmatTFAR, a)
areaTPC = trapz(gofmatTFTPC,a)

print("Area under the curve Surface to Volume=", areaSV)
print("Area under the curve Aspect Ratio=", areaAR)
print("Area under the curve Two Point Correlation=", areaTPC)





from sklearn import metrics
labels_true=tfnewdata200replicaslables
# Compute DBSCAN
mat=NorTF200replicasTPC
db = DBSCAN(eps=a[24], metric='precomputed').fit(mat)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"    % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(mat, labels))

LABEL_COLOR_MAP = {-1:'#35978f',
                   0 :'#d73027',
                   1 :'#fc8d59',
                   2:'#fee090',
                   3:'#762a83',
                   4:'#998ec3',
                   5:'#542788',
                   6:'#7fbf7b',
                   7:'#1b7837',
                   8:'#af8dc3',
                   9:'#b35806',
                   10:'#80cdc1',
                   
                   }




label_color = [LABEL_COLOR_MAP[l] for l in labels]
#plt.scatter(x, y, c=label_color)

True_label_color = [LABEL_COLOR_MAP[l] for l in labels_true]
plt.scatter(X_transformed[:,0], X_transformed[:,1],c =True_label_color)
plt.title('MDS projection with true lables')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')



plt.scatter(X_transformed[:,0], X_transformed[:,1],c =label_color)
plt.title('DBSCAN Clustering results Two point coorelation')
#plt.title('DBSCAN Clustering results Aspect Ratio')
#plt.title('DBSCAN Clustering results Surface to volume')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
cmap=('label_color')


markers_map={-1:'X',
                   0 :'v',
                   1 :'^',
                   2:'*',
                   3:'d',
                   4:'X',
                   5:'s',
                   }

markers= [markers_map[l] for l in labels]
for i in range(len(X_transformed)):
    plt.scatter(X_transformed[i,0], X_transformed[i,1],c =True_label_color[i],marker=markers[i])
    plt.title('DBSCAN Clustering results Two point coorelation')
    #plt.title('DBSCAN Clustering results Aspect Ratio')
    #plt.title('DBSCAN Clustering results Surface to volume')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
















#######################EAS NEW DATA######################
EASD3VS=np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio3",cosine=True))
EASD4VS=np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio4",cosine=True))


EASD3AR=np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio3", signature_function='shape_ratio_sig' , visualize_graphs=False,cosine=True))
EASD4AR=np.array(compute_distance(r"C:/Users/parth/Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\ExtremeAspectRatio4", signature_function='shape_ratio_sig' , visualize_graphs=False,cosine=True))

EASD3VSExpanded=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASExpandeddata2",cosine=True))
EASD4VSExpanded=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\New folder", visualize_graphs=True))


EASD3ARExpanded=np.array(compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Expanded Data\EASExpandeddata2", signature_function='shape_ratio_sig' , visualize_graphs=False,cosine=True))
EASD4ARExpanded=(compute_distance(r"C:\Users\parth\Desktop\New folder", signature_function='shape_ratio_sig' , visualize_graphs=True,cosine=True))




import seaborn as sns
sns.set(font_scale=1.4)

sns.clustermap(EASD3VS)
sns.clustermap(EASD3AR)


sns.clustermap(EASD4VS)
sns.clustermap(EASD4AR)


sns.clustermap(EASD3VSExpanded)
sns.clustermap(EASD3ARExpanded)


sns.clustermap(EASD4VSExpanded)
sns.clustermap(EASD4ARExpanded)




from upgma import generate_upgma

generate_upgma(Distance)
#
#import seaborn
#seaborn.heatmap(Distance)


####################################Distancematrix####################################

import seaborn as sns
sns.set(font_scale=1.4)

sns.clustermap(DistanceTFSurfacetoVolume)
sns.clustermap(DistanceMASSurfacetoVolume)
sns.clustermap(DistanceEASSurfacetoVolume)


sns.clustermap(DistanceTFAspectRatio)
sns.clustermap(DistanceMASAspectRatio)
sns.clustermap(DistanceEASAspectRatio)


sns.clustermap(DistanceTFSurfacetoVolumeExpanded)
sns.clustermap(DistanceMASSurfacetoVolumeExpanded)
sns.clustermap(DistanceEASSurfacetoVolumeExpanded)



sns.clustermap(DistanceTFAspectRatioExpanded)
sns.clustermap(DistanceMASAspectRatioExpanded)
sns.clustermap(DistanceEASAspectRatioExpanded)


sns.clustermap(TPCDistanceEAS2)

sns.clustermap(TPCDistanceEAS3expanded)


sns.clustermap(DVS)
sns.clustermap(DAS)
sns.clustermap(DFR)


sns.clustermap(DVSF)
sns.clustermap(DASF)
sns.clustermap(DFRF)


sns.clustermap(Dfractalvs)
sns.clustermap(DfractalAR)
sns.clustermap(DfractalF)


sns.heatmap(DistanceTFAR100replicasELayered)


################################################## Explained Variance####################





Distance=TPCDistanceEAS200replicas

from sklearn.manifold import MDS
stress = []
for i in range(1, 10):
    mds_sklearn = MDS(n_components=i, dissimilarity='precomputed')
    # Apply MDS
    pts = mds_sklearn.fit_transform(Distance[:])
    # Retrieve the stress value
    stress.append(mds_sklearn.stress_)
# Plot stress vs. n_components    
plt.plot(range(1, 10), stress)
plt.xticks(range(1, 10, 2))
plt.title('stress')
plt.xlabel('n_components')
plt.ylabel('stress')
plt.show()

















from SaveImage import compute_distance
Images=compute_distance(r'C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\spinodial Decomposotion\sd4x4\ImageFiles')

from two_point_correlation import compute_distance
D=compute_distance(r"C:\Users\parth\Desktop\MicrostructureDataGeneration\data2share\data2share\Composites\Data\Changed Volume fractions\200x800\200replicas\LayeredImages\Test order")



# #####SDTHINFILMS#################

# plt.scatter(X_transformed[0:5,0],X_transformed[0:5,1] )
# plt.scatter(X_transformed[5:10,0],X_transformed[5:10,1] )
# plt.scatter(X_transformed[10:15,0],X_transformed[10:15,1] )
# plt.scatter(X_transformed[15:20,0],X_transformed[15:20,1] )
# plt.scatter(X_transformed[20:25,0],X_transformed[20:25,1] )
# plt.scatter(X_transformed[25:30,0],X_transformed[25:30,1] )
# plt.scatter(X_transformed[30:35,0],X_transformed[30:35,1] )
# plt.scatter(X_transformed[35:40,0],X_transformed[35:40,1] )
# plt.scatter(X_transformed[40:45,0],X_transformed[40:45,1] )
# plt.scatter(X_transformed[45:50,0],X_transformed[45:50,1] )
# plt.scatter(X_transformed[50:55,0],X_transformed[50:55,1] )
# plt.scatter(X_transformed[55:60,0],X_transformed[55:60,1] )
# plt.scatter(X_transformed[60:65,0],X_transformed[60:65,1] )
# plt.scatter(X_transformed[65:70,0],X_transformed[65:70,1] )
# plt.scatter(X_transformed[70:75,0],X_transformed[70:75,1] )
# plt.scatter(X_transformed[75:80,0],X_transformed[75:80,1] )
# plt.scatter(X_transformed[80:85,0],X_transformed[80:85,1] )
# plt.scatter(X_transformed[85:90,0],X_transformed[85:90,1] )



