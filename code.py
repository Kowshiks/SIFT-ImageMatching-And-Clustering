import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


if sys.argv[1] == "part1":

    def two_image_func(image1,image2):

        # ORB feature function for 500 features
        orb = cv2.ORB_create(nfeatures=500)

        # detect keypoints and descriptors for both the images.
        (keypoints_1, descriptors_1) = orb.detectAndCompute(image1, None)
        (keypoints_2, descriptors_2) = orb.detectAndCompute(image2, None)

        match = []

        # Looping through the first source image
        for i in range(0, len(keypoints_1)):

            dist = {}

            # Looping through all the keypoints of the second image
            for j in range(0, len(keypoints_2)):

                # Finding the hamming distance with comparing 2 descriptors from 2 separate image
                norm_dist = cv2.norm( descriptors_1[i], descriptors_2[j], cv2.NORM_HAMMING)
                
                dist[keypoints_2[j].pt] = norm_dist

            # The matching points are sorted according to the hamming distance
            sorted_dict = dict(sorted(dist.items(), key=lambda x:x[1]))

            # Top 2 matching points are extracted
            pairs = {k: sorted_dict[k] for k in list(sorted_dict)[:2]}

            first_index, second_index = pairs.values()

            # A threshold of 0.8 is provided and also the hamming distance of first matching point should be less than 50
            if (first_index/second_index < 0.8) and (first_index < 50):

                match.append((keypoints_1[i].pt,next(iter(pairs))))

        return match


    # This function clusters all the distance count matrix
    def clustering_algorithm(X, k):

        model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
        model.fit(X)
        labels = model.labels_

        return labels

    # This function finds the pairwise accuracy for all the test images
    def find_accuracy(labels,file):

        N = len(file)

        tp = 0
        tn = 0

        for i in range(len(labels)):
            master_name = file[i].split("_")[0]
            for j in range(len(labels)):
                sub_name = file[j].split("_")[0]
                if i!=j:
                    if master_name == sub_name:
                        if labels[i] == labels[j]:
                            tp+=1
                    else:
                        if labels[i] != labels[j]:
                            tn+=1

        accuracy = (tn+tp)/(N*(N-1))
        return accuracy


    # This function writes the clustered images into a seperate file 
    def write_file(labels,file,output_file):

        d = {}
        k = 10

        for i in range(k):
            d[i] = []

        for i in range(len(labels)):
            
            d[labels[i]].append(i)


        final_output = []

        for value in d.values():
            
            output = []
            
            for each in value:
                
                output.append(file[each])
            
            final_output.append(output)

        with open(output_file,"w",encoding="utf-8") as fo:
            fo.write('\n'.join([' '.join(i) for i in final_output]))


    # This function is used to cluster the images usinf BFMatcher
    def clustering(file, k, output_file):

        grayscale  = []

        for each_file in file:

            grayscale.append(cv2.imread(each_file, cv2.IMREAD_GRAYSCALE))
        
        orb = cv2.SIFT_create()

        keypoint_list = []
        descriptor_list = []

        for gray_image in grayscale:
            
            (keypoints, descriptors) = orb.detectAndCompute(gray_image, None)
            keypoint_list.append(keypoints)
            descriptor_list.append(descriptors)

        
        bf = cv2.BFMatcher()

        total_matches = []

        # The keypoints of the source images are looped through
        for index_1 in range(len(keypoint_list)):
            
            matches_list = []
            
            for index_2 in range(len(keypoint_list)):
                
                
                if(index_1 != index_2):
                    
                    # Using KNN match the top 2 matching points are filteres
                    matches = bf.knnMatch(descriptor_list[index_1],descriptor_list[index_2],k=2)
                    # Referenced from KNN SIFT CV Documentation "https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html"
                    match_result = []
                    
                    # This for loop extracts the matching point by comparing best match and the second best match using the threshold
                    for first,second in matches:
                        
                        if first.distance < 0.8*second.distance and first.distance < 200:

                            match_result.append(first)

                    # Implementation of the distance number matrix       
                    if match_result:
                        result = 100/len(match_result)*2
                    
                    else:
                        result = 100
                            
                    matches_list.append(result)      
                          
                else:
                    
                    matches_list.append(0)
                    
            total_matches.append(matches_list)

        # Here symmertix matix is made by adding up all the source images with respect to the rest of the images                    
        final = np.array(total_matches) +  np.array(total_matches).T   

        total_matches = list(final)

        X = np.array(total_matches)

        # Clustering algorithm function is called here.
        label = clustering_algorithm(X, k)

        #accuracy = find_accuracy(label,file)

        #print("Pairwise Clustering Accuracy : ",accuracy)

        # Function to write the output file.
        write_file(label, file, output_file)

    #img_1 = cv2.imread("part1-images/bigben_6.jpg", cv2.IMREAD_GRAYSCALE)

    #img_2 = cv2.imread("part1-images/bigben_8.jpg", cv2.IMREAD_GRAYSCALE)


    #two_image_func(img_1,img_2)
                
    # K clusters
    k = int(sys.argv[2])

    filename = sys.argv[3:-1]

    output_file = sys.argv[-1]

    match_points = clustering(filename, k, output_file)


