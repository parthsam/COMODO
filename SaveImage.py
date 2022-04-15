# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:50:05 2021

@author: parth
"""

#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from matplotlib import pyplot as plt
from skimage import measure
from skimage.future import graph
import networkx as nx
import sys
import scipy.misc
import imageio
from scipy.spatial import distance as dist



def compute_distance(image_set_directory, 
                     signature_function='surface_volume_ratio_sig', 
                     visualize_graphs=False,
                     weighted=False,
                     cosine=False):
    image_vectors = []
    components = []
    rags = []
    rags_dict = {}

    # x=199
    # y=99
    # # generate images from plt files
    # source_directory = Path(image_set_directory)
    # for idx, image in enumerate(source_directory.iterdir()):
    #       # read the image file and process it
    #       img_data = np.genfromtxt(image)
    #       img = img_data.reshape(100 ,200 )
    #       plt.imsave(f'{image}.jpg', img, cmap='gray')

    # `image_set_directory` can either be a string containing path to the files or
    # a list of strings containing path to multiple files
    if isinstance(image_set_directory, str):
        sorted_img_files = sorted([i for i in Path(image_set_directory).iterdir()])
    elif isinstance(image_set_directory, list):
        sorted_img_files = []
        for directory in image_set_directory:
            sorted_directory_files = sorted([i for i in Path(directory).iterdir()], key=lambda x: int(str(x).split("/")[-1].split(".")[-2]))
            sorted_img_files += sorted_directory_files


    for p in sorted_img_files:
        if p.is_file():
            print(p)

            # read image data and convert the image to grayscale
            # img = io.imread(p)
            # if the image has an alpha (tansparency) chanel remove it
            # if img.shape[-1] == 4:
            #     img = rgba2rgb(img)
            # img = rgb2gray(img)

            # read raw bitwise data 
            img_data = np.genfromtxt(p)
            img = img_data.reshape((400, 400))
            #img = np.where(img > 0.5, 1, 0)
            target_dir = Path(image_set_directory)/"images"
        
            # create if path does not exist
            target_dir.mkdir(parents=True, exist_ok=True)
            
            plt.imsave(target_dir/(f"{p.stem}.png"), img,cmap='gray')


            # generate region adjacency graph
            rag = generate_region_adjacency_graph(img, signature_function)
            rags.append(rag)
            rags_dict[p.stem] = rag
            # rag, component = generate_region_adjacency_graph(img, signature_function)
            # components.append(component)

            # visualize graphs if required
            if visualize_graphs:
                generate_graph_visualization_images([rag], p.stem)
            # generate bfs vector
            vector = generate_bfs_vector(rag)
            image_vectors.append(vector)

    # generate bfs vector
    distances = []
    for vector_1 in image_vectors:
        distance_row = []
        for vector_2 in image_vectors:
            # make the 2 vectors of equal length
            vector_pair = [vector_1, vector_2]
            padded_vectors = generate_padded_vectors_reversable(vector_pair)
            if cosine:
                v1, v2 = padded_vectors
                abs_padded_vectors = [[np.abs(i) for i in v1], [np.abs(j) for j in v2]]
                distance = dist.cosine(*padded_vectors)
            else:
                # compute the euclidean distance of the 2 vectors
                if weighted:
                    distance = weightedL2(padded_vectors[0], padded_vectors[1])
                else:
                    distance = np.linalg.norm(padded_vectors[0] - padded_vectors[1])

            distance_row.append(distance)
        distances.append(distance_row)

    return distances
    # return generate_padded_vectors_reversable(image_vectors)
    # return components

def weightedL2(vector_1, vector_2):
    diff = vector_1 - vector_2
    weight_array = [damping_function(i) for i in np.arange(len(diff))]
    return np.sqrt((weight_array*diff*diff).sum())

def damping_function(x):
    return 10*np.exp(-0.05*x)

def generate_region_adjacency_graph(image, signature_function):
    """
    Create a region adjacency graph from a given image
    Args:
        image (ndarray):
            grayscale image
        signature_function (int):
            function to be used to calculate the signal of each
            component/blob
    Return:
        rag (RAG):
            region adjacency graph
    """
    # identify neighbouring pixels with the same pixel value,
    # assign them labels and split them
    components, binary_image, label_image = extract_components(image, allow_blank_images=True, return_images=True)

    # make sure number of components = number of unique labels
    # make sure no component is being pruned
    assert len(components) == len(np.unique(label_image)), "Total components != Total labels"

    # generate the Region Adjacency Graph
    rag = graph.rag_mean_color(binary_image, label_image)

    # calculate the signature of each component
    sigs = []
    for component in components:
        sig = apply_signatures(component, signature_function, allow_empty_sig=True)
        sigs.append(sig[0])

    # make sure number of signatures = number of components
    # Ensure no sig value is getting pruned
    assert len(components) == len(sigs), "Total signatures != Total components"

    # delete components whose signature is None
    # those components need to be pruned
    for idx, sig in enumerate(sigs):
        if not sig:
            rag.remove_node(idx + 1)

    # create a pruned component list that only 
    # has components with valid signatures
    pruned_components = []
    for idx, sig in enumerate(sigs):
        if sig:
            pruned_components.append(components[idx])

    # add signatures as node weights
    for idx, sig in enumerate(sigs):
        # only add signature when signature is not None
        # None signature means the component needs to 
        # be ignored
        if sig:
            rag.nodes[idx + 1]['weight'] = sig

            # remove unwanted data in the nodes
            del rag.nodes[idx + 1]['total color']

    # remove edge weights since they are not required
    for edge in rag.edges:
        node_1, node_2 = edge
        del rag[node_1][node_2]['weight']

    # inform user about the components that were neglected
    total_components_pruned = len(components) - len(pruned_components)
    print(f"Pruned {total_components_pruned} component(s)")

    # return rag, pruned_components
    # return rag, components
    return rag

def generate_graph_visualization_images(graphs, filename, combined=True):
  """
  Save graph visualizations as images.
  Args:
      graphs: list of graphs to be visualized
      combined: if False, directory name will
                be determined based on index number 
                of the graph. Default=True.
  """
  # trajectory_name, trajectory_index = trajectory_name.split("_")
  # target_dir = Path("/home/namit/codes/Entropy-Isomap/outputs/constDt-5replicas-noPer-4x4_graphs")/trajectory_name/(f"{trajectory_index}.png")
  # if target_dir.exists():
    # return None

  root_nodes = []
  for graph_num,gg in enumerate(graphs):
      
      fig,axes = plt.subplots()

      # import ipdb; ipdb.set_trace()
      
      # setting node size
      node_size = [i[1]['pixel count'] for i in gg.nodes(data=True)]
      sum_node_size = sum(node_size)
      node_size_normalized = [(i/sum_node_size)*5000 for i in node_size]
      
      # setting node color
      node_color = []
      for i in gg.nodes(data=True):
          current_color = i[1]['mean color'][0]
          if current_color == 1:
              # this is white
              # set to light grey
              node_color.append(np.array([0.7,0.7,0.7]))
          elif current_color == 0:
              # this is black
              # set to dark grey
              node_color.append(np.array([0.3,0.3,0.3]))
          else:
              # this should never happen
              print("Unknown color of node.")
      
      # setting node label
      node_labels = {}
      for index, size in enumerate(node_size):
          node_labels[index+1] = f"{size}"
          # node_labels[index+1] = f"{size} ({index+1})"
          
      # setting node edge colors
      edgecolors = ['k']*len(node_color)
      root_node = get_max_degree_node(gg)
      # print(f"{graph_num} - {root_node}")
      try:
          edgecolors[root_node-1] = 'r'
      except:
          nx.draw_kamada_kawai(graph)
          return None
      root_nodes.append(root_node)
      
      
      # create the graph and save it
      nx.draw_kamada_kawai(gg, 
                            node_color  = node_color,
                            edgecolors  = edgecolors,
                            labels      = node_labels,
                            with_labels = True,
                            ax          = axes)
      
      # target_dir = Path("/home/namit/codes/Entropy-Isomap/outputs/constDt-5replicas-noPer-4x4_graphs")/trajectory_name
      target_dir = Path("./")/"graphs"
  
      # create if path does not exist
      target_dir.mkdir(parents=True, exist_ok=True)
      
      # title = trajectory_name+f" #{graph_num%80} ({graph_num})"
      # plt.title(title, y=-0.1)
      
      plt.savefig(target_dir/(f"{filename}.png"))
      # plt.savefig(f'/home/namit/codes/meads/morphology-similarity/playground/Results/organic_morphology_graph_{m_idx}.pdf')

      print("generated graph file: ", target_dir/(f"{filename}.png"))
      
      plt.cla()
      plt.close()
  return root_nodes

def generate_bfs_vector(graph, return_traversal_order=False):
    """
    Vecotorize a given graph using the priority BFS algorithm
    Args:
        graph (networkx.Graph):
            The input graph to use
        return_traversal_order (bool):
            Whether to return traversal order of the nodes.
            (default=False)
    Returns:
        vector (list):
            vector representation of the graph
        traversal_order (list):
            A list containing the indices of the nodes in the order they
            were traversed. Only returned when return_traversal_order is True.
    """
    # determine the root node
    root = get_max_degree_node(graph)

    # generate BFS vector
    return priority_bfs(graph, root, return_traversal_order=return_traversal_order)

def generate_padded_vectors(vectors):
    """
    Implements layerwise padding.
    Note this function only pads any two vectors at a time by aligning
    all the white nodes one under the other and filing empty spaces
    with empty nodes in between.
    Args:
        vectors (list): A list of length 2 containing the 2 vectors to be padded
    Returns:
        padded_vectors (list): A list of length 2 containing the 2 padded vectors
    """

    split_vectors = []

    for vector in vectors:
        # split the nodes based on sign
        split_indices = []

        for index,d in enumerate(vector):
            if d >= 0:
                split_indices.extend([index, index+1])

        split_vector = np.split(vector, split_indices)
        split_vectors.append(split_vector)


    # get the maximum length of split at each
    # position for all the vectors
    max_split_length = {}

    for split_vector in split_vectors:
        for index,split in enumerate(split_vector):
            max_split_length[index] = max([len(split), max_split_length.get(index, 0)])


    # pad all the splits to their
    # respective max lengths
    padded_split_vectors = []
    for split_vector in split_vectors:

        padded_split_vector = []
        for index,split in enumerate(split_vector):
            padded_split = front_pad(split, max_split_length[index])
            padded_split_vector.append(padded_split)

        padded_split_vectors.append(padded_split_vector)


    # merge all splits into single vector
    merged_vectors = []
    for padded_split_vector in padded_split_vectors:
        merged_vector = np.concatenate(padded_split_vector)
        merged_vectors.append(merged_vector)


    # over all frontpad to compensate for
    # different number of layers in each graph
    max_dimension = max(map(len, merged_vectors))

    padded_vectors = []
    for merged_vector in merged_vectors:

        padded_vector = front_pad(merged_vector, max_dimension)
        padded_vectors.append(padded_vector)


    # make all vector dimension magnitudes
    # positive irrespective of color
    # positive_vectors = []
    # for padded_vector in padded_vectors:
    #   positive_vector = np.abs(padded_vector)
    #   positive_vectors.append(positive_vector)

    # write some tests maybe
    return padded_vectors

def extract_components(image, 
                       binarize=True, 
                       background=-1,
                       allow_blank_images=False,
                       return_images=False):
    """
    Extract morpholgoical components from an image.
    Arguments:
        image (ndarray): 
            A grayscale image 
        binarize (boolean): 
            Flag to binarize the image before extraction.Defaults to
            True.
        background (int): 
            Pixels of this value will be considered background and will
            not be labeled. By default every pixel is considered for
            labeling. Set to -1 for no background. (default=-1)
        allow_blank_images (boolean):
            Whether a blank image should be considered as a single
            component. (default=False) 
        return_images (boolean): 
            Wheter to return the labeled image & binary
            image.(default=False)
    Returns:
        components (list): 
            A list of component images with the same shape as the input
            image.
        image (ndarray): 
            Original image (and binarized if binarize=True). Only
            returned when return_images is set to True.
        labeled_sample (ndarray): 
            A labelled image where all connected ,regions are assigned
            the same integer value. Only returned when return_images is
            set to True.
    """
    components = []
    if binarize:
        image = (image > 0.5).astype(int)

    labeled_sample = measure.label(image, background=background)

    for label in np.unique(labeled_sample):
        # extract companent into a separate image
        component = (labeled_sample == label).astype(np.float64)
        
        if not allow_blank_images:
            if (component == 0).all():
                continue 
                
        components.append(component)

    # remove the first background component if background pixels needs
    # to be neglected. 
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#label
    if background >= 0:
        components = components[1:]
    
    if return_images:
        return components, image, labeled_sample
    else:
        return components

def apply_signatures(image, sig_funcs, allow_empty_sig=False):
    """
    Applies the provided signature functions to the given image.
    Arguments:
        image: An image, represented as a 2D Numpy array.
        sig_funcs: List of signature extraction functions.
        allow_empty_sig: If signature values of zero should be allowed
                         (default=False)
    Returns:
        A list of signatures for the given image.
    Raises:
        AssertionError: All signatures returned by the extractors need
        to be non-empty. Only when allow_empty_sig is False.
    """
    if isinstance(sig_funcs, str):
        # For convenience, we can pass in a single signature function.
        # This converts it into a list, with it being the only element.
        sig_funcs = [sig_funcs]
    sigs = []
    for sig_func in sig_funcs:
        sig = eval(sig_func+"(image)")
        if not allow_empty_sig:
            assert len(sig) > 0, 'Signatures need to be non-empty.'
        sigs.append(sig)
    sigs = np.array(sigs).T
    return sigs

def surface_volume_ratio_sig(component):
    """The surface to volume (perimeter to area) ratio"""
    perimeter = measure.perimeter(component)
    area = np.sum(component == 1)
    # Note: We are guranteed to have at least 1 pixel of value 1
    # the perimeter of a single pixel is also 1
    if perimeter == 0:
        return None
    return perimeter/area

def shape_ratio_sig(component):
    """The ratio of the width of the component to it's height"""
    # The component is the only part of the full image that is 1
    # we need to generate a bounding box around the component 
    # measure the ratio of its height and width

    # convert component array into integer array
    component = component.astype(np.int64)

    # generate region properties of the component
    regions = measure.regionprops(component)

    # the component image will have only one component
    min_row, \
    min_col, \
    max_row, \
    max_col = regions[0]['BoundingBox']
    
    lengthXdirection= (max_row-min_row)
    lengthYdirection= (max_col-min_col)
    minlength=min(lengthXdirection,lengthYdirection)
    maxlength=max(lengthXdirection,lengthYdirection)
    return(minlength/maxlength)

def fractal_dimension_sig(component):
    """The fractal dimension of the component"""
    if np.isnan(fractal_dimension(component)):
        return None
    area = np.sum(component == 1)
    if area < 100:
        return None
    return fractal_dimension(component)

def get_max_degree_node(graph):
    """
    Determine the node with the maximum degree irrespective of its color
    Args:
        graph (networkx.Graph): 
            The input graph to use.
    Returns:
        max_degree_node (int): 
            index of the node with the maximum degree 
    """

    nodes = list(graph.nodes(data=True))
    max_degree_node = nodes[0][0]

    # Iterate over the nodes and find the most connected node
    for node in nodes:
        if graph.degree[node[0]] > graph.degree[max_degree_node]:
            max_degree_node = node[0]
        elif graph.degree[node[0]] == graph.degree[max_degree_node]:
            # settle a tie by choosing the node with greater area
            if node[1]['pixel count'] > \
                graph.nodes(data=True)[max_degree_node]['pixel count']:
                max_degree_node = node[0]

    return max_degree_node

def priority_bfs(graph, root, return_traversal_order=False):
    """
    Implementation of the priority BFS algorithm
    Args:
        graph (networkx.Graph): 
            The input graph to use.
        root (int):
            index of the node to be considered as the root
        return_traversal_order (bool): 
            Whether to return traversal order of the nodes.
            (default=False)
    Returns:
        vector (list): 
            vector representation of the graph
        traversal_order (list): 
            A list containing the indices of the nodes in the order they
            were traversed. Only returned when return_traversal_order is
            True.
        
    """
    vector  = []
    visited = []
    queue   = []

    # Queue element storage format 
    # [ (<node>, <node_signature>) , (<node>, <node_signature>), ... ]
    queue.append((root,graph.nodes[root]['weight']))

    while queue:

        # Step A: Dequeue it to the vector
        current_node_index, current_node_signature = queue.pop(0)
        current_node_color = graph.nodes[current_node_index]['mean color'][0]
        visited.append(current_node_index)


        # Step B: Append it to the vector
        vector.append(get_node_color_sign(current_node_color) *
                      current_node_signature)


        # Step C: Get all of elements children
        current_node_neighbors = []
        for neighbor in graph.neighbors(current_node_index):
            current_node_neighbors.append(
                (neighbor, graph.nodes[neighbor]['weight']))


        # Step D: Sort them by their signature and enqueue them
        current_node_neighbors.sort(key = lambda x: x[1])
        # enqueueing - make sure that node has not been visited first
        # althugh that should not happen since the graph is always
        # acyclic
        for neighbor in current_node_neighbors:
            if neighbor[0] not in visited:
                queue.append(neighbor)

    vector = np.array(vector)

    if return_traversal_order:
        return vector, visited
    else:
        return vector

def get_node_color_sign(node_color):
    """
    Returns -1 for black color (node_color = 0) and 
    1 for white (node_color = 1 or 255)
    Args:
        node_color (int): 
            node color value should be 0, 1 or 255
    
    Returns:
        sign (int): -1 or 1 based in pixel value
    Raises:
        AssertionError: 
            node_color needs to be either black or white.
    """
    assert node_color in [0,1,255], "node_color can only be 0, 1 or 255"

    if node_color < 2:
        # when node color value is 0 or 1 that means it is a black node
        # black node is -1
        return ((-1) ** (node_color+1))
    else:
        # 255 is always white
        # white node is 1
        return 1

def generate_padded_vectors_reversable(vectors):
    """
    Implements layerwise padding. 
    Note this function only pads any two vectors at a time by aligning 
    all the white nodes one under the other and filing empty spaces
    with empty nodes in between. 
    NOTE:
    This is a modified version of the generate_padded_vectors function. 
    This function instead of aligning all white peaks one under the other
    aligns all the continous chunks of elements one under the other by 
    making them of the same length. 
    Args:
        vectors (list): A list of length 2 containing the 2 vectors to be padded
    Returns:
        padded_vectors (list): A list of length 2 containing the 2 padded vectors
    """
     
    split_vectors = []

    for vector in vectors:
        # split the nodes into continous chunks of values having the same sign
        # same sign means same color
        split_vector = []

        previous = vector[0]
        sub_split_vector = []

        for element in vector:
            # component's signature value should always be non-zero
            assert element != 0, 'Element with 0 signature found'

            if element*previous > 0:
                # current and the previous element have the same sign
                sub_split_vector.append(element)
            else:
                # current and the previous do not have the same sign
                split_vector.append(sub_split_vector)
                sub_split_vector = [element]

            previous = element

        # appending the last sub split 
        # since there was no sign change to trigger the sign channge
        split_vector.append(sub_split_vector)

        split_vectors.append(split_vector)

    # for vector in vectors:
    #     # split the nodes based on sign
    #     split_indices = []

    #     for index,d in enumerate(vector):
    #         if d >= 0:
    #             split_indices.extend([index, index+1])

    #     split_vector = np.split(vector, split_indices)
    #     split_vectors.append(split_vector)


    # print([print(i) for i in split_vectors])



    # get the maximum length of split at each 
    # position for all the vectors
    max_split_length = {}

    for split_vector in split_vectors:
        for index, split in enumerate(split_vector):
            max_split_length[index] = max([len(split), max_split_length.get(index, 0)])



    # pad all the splits to their 
    # respective max lengths
    padded_split_vectors = []
    for split_vector in split_vectors:

        padded_split_vector = []
        for index,split in enumerate(split_vector):

            padded_split = front_pad(split, max_split_length[index])
            padded_split_vector.append(padded_split)

        padded_split_vectors.append(padded_split_vector)



    # Merge all splits into single vector
    merged_vectors = []
    for padded_split_vector in padded_split_vectors:
        merged_vector = np.concatenate(padded_split_vector)
        merged_vectors.append(merged_vector)


    # over all frontpad to compensate for
    # different number of layers in each graph
    max_dimension = max(map(len, merged_vectors))

    padded_vectors = []
    for merged_vector in merged_vectors:

        padded_vector = front_pad(merged_vector, max_dimension)
        padded_vectors.append(padded_vector)


    # make all vector dimension magnitudes 
    # positive irrespective of color
    # positive_vectors = []
    # for padded_vector in padded_vectors:

    #   positive_vector = np.abs(padded_vector)
    #   positive_vectors.append(positive_vector)

    # write some tests maybe
    return padded_vectors

def front_pad(vector, max_dimension):
    """
    Add zeroes in front of the given vector
    """
    # make np array if not
    if not isinstance(vector, np.ndarray): vector=np.array(vector)

    return np.pad(vector, (0,max_dimension-len(vector)))

def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# if len(sys.argv) != 5:
#     print("usage: compute_distance.py <x> <y> <path_to_morphology_set_directory> <path_to_query_image>")
#     sys.exit(-1)

# print(compute_distance(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))