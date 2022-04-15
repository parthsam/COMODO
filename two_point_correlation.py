#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage import measure
from skimage.future import graph
import networkx as nx
import sys
import scipy.misc
import imageio
from pymks import PrimitiveTransformer, TwoPointCorrelation
from sklearn.pipeline import Pipeline



def compute_distance(image_set_directory):
    image_arrays = []

    sorted_img_files = sorted([i for i in Path(image_set_directory).iterdir()])
    for p in sorted_img_files:
        if p.is_file():
            print(p)

            # read image data and convert the image to grayscale
            img = io.imread(p)
            img = rgb2gray(img)
            img = (img > 0.5).astype(int)


            # read raw bitwise data 
            # img_data = np.genfromtxt(p)
            # img = img_data.reshape((100, 400))

            image_arrays.append(img)

    # 2pt correlation 
    distance_matrix = np.empty((len(image_arrays), len(image_arrays)))
    twopt_correlations = []

    for m1_idx, morphology_file_1 in enumerate(image_arrays):
        binary_morphology_1 = morphology_file_1
        
        # reshape the image based on requirements
        binary_morphology_1 = binary_morphology_1.reshape((1, *binary_morphology_1.shape))

        # create the 2 point correlation pipeline
        model = Pipeline(steps=[
                    ('discretize', PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0)),
                    ('correlations', TwoPointCorrelation(
                        periodic_boundary=True,
                        cutoff=400,
                        correlations=[[0, 0]]
                    ))
                ])

        # generate 2 point correlation
        binary_morphology_1_stats = model.transform(binary_morphology_1).persist()
        binary_morphology_1_stats = binary_morphology_1_stats[0, :, :, :]
        a=binary_morphology_1_stats.compute()
       # print(np.size(a))
        
        fname=(p.stem + '.txt')
        print(fname)
       # np.savetxt(fname,a[:,:,0])
 
        # print("#correlations: ", binary_morphology_1_stats.shape[-1])
        for i in range(binary_morphology_1_stats.shape[-1]):
            twopt_correlations.append(binary_morphology_1_stats[:, :, i].flatten())


    array_len = len(twopt_correlations)
   # print(twopt_correlations)
    d = np.empty((array_len, array_len))
    for i in range(array_len):
        for j in range(array_len):
            #print(twopt_correlations[i])
            d[i,j] =  np.linalg.norm(twopt_correlations[i] - twopt_correlations[j])

    return d