#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import time
from os import listdir
from os.path import isfile, join

import bigfish.detection as detection
import bigfish.stack as stack
import numpy as np
import tifffile
from numpy import argmax, nanmax, unravel_index
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform


from sklearn.cluster import OPTICS, cluster_optics_dbscan
import math

# todo re-order function
#%%


def spot_detection_for_clustering(sigma, rna_path, path_output_segmentaton,
                                  threshold_input=None,
                                  output_file="detected_spot_3d/",
                                  min_distance=(3, 3, 3),
                                  local_detection = True,
                                  diam=20,
                                  scale_xy=0.103,
                                  scale_z=0.300,
                                  min_cos_tetha=0.80,
                                  order=5,
                                  test_mode=False
                                  ):
    """
    save arry of coordiante of detected spots
    Args:
        sigma (list):
        rna_path (list): list of rna path
        path_output_segmentaton (str):
        threshold_input (dict): dictionary {image_name: hardcoded threshold}
                            if None the threshold is computed automatically by bigfish
        output_file ():
        min_distance ():

    Returns:
        dico_threshold (dict) : {image_name: hardcoded threshold use}
    """
    dico_threshold = {}
    onlyfiles = [f for f in listdir(path_output_segmentaton) if isfile(join(path_output_segmentaton, f))
                 and f[-1] == "f"]
    onlyfiles = [onlyfiles[i][14:] for i in range(len(onlyfiles))]
    print(onlyfiles)
    for index_path in range(len(rna_path)):
        path = rna_path[index_path]
        for file_index in range(len(onlyfiles)):
            t = time.time()
            rna = tifffile.imread(path + onlyfiles[file_index])

            if local_detection:
                print("local_detection")
                segmentation_mask = tifffile.imread(path_output_segmentaton +"dapi_maskdapi_" + onlyfiles[file_index])
                spots = detection_with_segmentation(rna = rna,
                                         sigma = sigma,
                                          min_distance = min_distance,
                                          segmentation_mask = segmentation_mask,
                                          diam=diam,
                                          scale_xy=scale_xy,
                                          scale_z=scale_z,
                                          min_cos_tetha=min_cos_tetha,
                                          order=order,
                                          test_mode=test_mode)
                threshold = None
            else:
                print(sigma)
                rna_log = stack.log_filter(rna, sigma)  # , float_out)
                # local maximum detection
                mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
                if threshold_input is not None and onlyfiles[file_index] in threshold_input:
                    threshold = threshold_input[onlyfiles[file_index]]
                    rna_log = stack.log_filter(rna, sigma)  # , float_out = False)
                    print("manuel threshold")
                else:
                    threshold = detection.automated_threshold_setting(rna_log, mask)
                print(threshold)
                spots, _ = detection.spots_thresholding(rna_log, mask, threshold)


            dico_threshold[onlyfiles[file_index]] = [threshold, len(spots)]
            np.save(output_file + path[-6:] + onlyfiles[file_index][:-5] + 'array.npy', spots)
            print(len(spots))
    return dico_threshold


def computer_optics_cluster(spots, eps=25, min_samples=10, min_cluster_size=10,
                            xi=0.05, scale=np.array([(300/103), 1, 1])):
    """
    Parameters
    apply OPTICS (Ordering Points To Identify the Clustering Structure)
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
    ----------
    spots
    eps
    min_samples
    min_cluster_size
    xi
    scale
    Returns label, arrays with the cluster index of each
    -------
    """

    try:
        clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=int(min_cluster_size))
        # Run the fit
        if len(scale) == 3 and len(spots[0]) == 3:
            print("rescale the clustering")
            print(len(spots))
            clust.fit(spots * scale)
        else:
            clust.fit(spots)        
        labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,  ordering=clust.ordering_, eps=eps)
        return labels
    except ValueError as e:
        print(e)
        return np.array([-1] * len(spots))
    
    
def cluster_over_nuclei_3D_convex_hull(labels, spots, masks, iou_threshold=0.5, scale=(300, 103, 103), max_spots=40000,
                                       min_nb_spots_per_cluster=3):
    """
    Compute the convex hull of each cluster plus cell classification
    if a cell is in the convex hull of a cluster of the input probe, it is considere as positive to this probe
    Args:
        labels ():  output of computer_optics_cluster
        spots (): array of coordiante of spots
        masks (): segmentation mask of nuclei
        iou_threshold ():
        scale ():
        max_spots ():

    Returns:

    """

    positive_cell = []
    positive_cluster = []
    negative_cluster = []
    mask_single_cell = (masks > 0)
    all_nuclei_coord = np.array(list(zip(*np.nonzero(mask_single_cell))))

    if len(spots) > max_spots:
        print(f'number of spots ({len(spots)}) superior to  max_spots {max_spots} '
              f'it migth be an error : return empty list')
        return positive_cell, positive_cluster, negative_cluster
    for cluster in range(np.max(labels)+1):
        print(cluster)
        cluster_spots = spots[labels == cluster]
        print(len(cluster_spots))
        print()
        if len(cluster_spots) <= min_nb_spots_per_cluster:
            continue
        t = time.time()
        try:
            convex_hull = Delaunay(cluster_spots)
        except Exception as e:
            print(e)
            continue
        cluster_spots[:, 0] = cluster_spots[:, 0] * scale[0]
        cluster_spots[:, 1] = cluster_spots[:, 1] * scale[1]  # be careful to rescal it only once
        cluster_spots[:, 2] = cluster_spots[:, 2] * scale[2]
        try:
            D = pdist(cluster_spots)
            D = squareform(D)
        except Exception as e:
            print(e)
            raise e
            #  return [], [], []
        longuest_distance, [I_row, I_col] = nanmax(D), unravel_index(argmax(D), D.shape)
        all_coord_bool = convex_hull.find_simplex(all_nuclei_coord) >= 0
        dapi_cordo = all_nuclei_coord.reshape(-1, 3)[all_coord_bool]
        # take only into account cell that ovellap the point cloud
        candidate_cells = np.sort(np.unique([masks[tuple(co)] for co in dapi_cordo]))
        for cs in candidate_cells:  # exclude zero is done before
            try:
                t = time.time()    
                mask_single_cell = (masks == cs)    
                cell_coord = np.array(list(zip(*np.nonzero(mask_single_cell))))
                overlap = np.sum(convex_hull.find_simplex(cell_coord) >= 0) / len(cell_coord) 
                if overlap > iou_threshold:  # c'est pas un iou just un threshold
                    # print((cluster, cs))
                    positive_cell.append(cs)
                    print(f"positive cell {cs}")
                    positive_cluster.append([cluster, overlap, ConvexHull(cluster_spots).volume,
                                             cs, ConvexHull(cluster_spots).area, longuest_distance, len(cluster_spots)])
                else:
                    if overlap > 0:
                        negative_cluster.append([cluster, overlap, ConvexHull(cluster_spots).volume,
                                                 cs, len(cluster_spots)])
                        print(f"negative_cluster {cs}")

            except Exception as e:
                    print(e)
        print(time.time()-t)
    return positive_cell, positive_cluster, negative_cluster

##################################################
## Spot detection based on cell positon plus noise removal
##################################################







def mean_cos_tetha(gy ,gx, z, yc, xc, order = 3):
    """
    todo add checking
    Args:
        gy (): gz, gy, gx = np.gradient(rna_gaus)
        gx ():
        z ():  z coordianate of the detected spots
        yc (): yc coordianate of the detected spots
        xc (): xc coordianate of the detected spots
        order (): number of pixel in xy away from the detected spot to take into account
    Returns:

    """
    import math
    list_cos_tetha = []
    for i in range(xc-order, xc+order+1):
        for j in range(yc-order, yc+order+1):
            if i - xc < 1 and j-yc < 1:
                continue
            if i < 0 or i > gx.shape[2]-1:
                continue
            if j < 0 or j > gx.shape[1]-1:
                continue
            vx = (xc - i)
            vy = (yc - j)

            cos_tetha = (gx[z, j, i]*vx + gy[z, j, i]*vy) / (np.sqrt(vx**2 +vy**2) * np.sqrt(gx[z, j, i]**2 + gy[z, j, i]**2) )
            if math.isnan(cos_tetha):
                continue
            list_cos_tetha.append(cos_tetha)
    return np.mean(list_cos_tetha)



### function to remove double detection


import itertools

def remove_double_detection(input_array,
            threshold = 0.3,
            scale_z_xy = np.array([0.300, 0.103, 0.103])):
    """

    Args:
        input_list (np.array):
        threshold (float): min distance between point in um
        scale_z_xy (np.array):voxel scale in um

    Returns: list of point without double detection

    """
    unique_tuple = [tuple(s) for s in input_array]
    unique_tuple = list(set((unique_tuple)))

    combos = itertools.combinations(unique_tuple, 2)
    points_to_remove = [list(point2)
                        for point1, point2 in combos
                        if np.linalg.norm(point1 * scale_z_xy  - point2 * scale_z_xy) < threshold]

    points_to_keep = [point for point in input_array if list(point) not in points_to_remove]
    return points_to_keep


### function that take fish signal, segmentation mask output the detected spots


from scipy import ndimage
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity
from tqdm import tqdm

def detection_with_segmentation(rna,
                                sigma,
                                min_distance,
                              segmentation_mask,
                              diam = 20,
                              scale_xy = 0.103,
                              scale_z = 0.300,
                              min_cos_tetha = 0.75,
                              order = 5,
                              test_mode = False,
                              threshold_merge_limit = 0.3):
    """

    Args:
        rna ():
        sigma ():
        min_distance ():
        segmentation_mask ():
        diam ():
        scale_xy ():
        scale_z ():
        min_cos_tetha ():
        order ():
        threshold_merge_limit (float): threshold below to detected point are considere the same

    Returns:

    """
    rna_log = stack.log_filter(rna, sigma)
    mask = detection.local_maximum_detection(rna_log, min_distance=min_distance)
    rna_gaus = ndimage.gaussian_filter(rna, sigma)

    list_of_nuc = np.unique(segmentation_mask)
    if 0 in list_of_nuc:
        list_of_nuc = list_of_nuc[1:]
    assert all(i >= 1 for i in list_of_nuc)

    all_spots = []
    pbar = tqdm(list_of_nuc)
    for mask_id in pbar:
        pbar.set_description(f"detecting rna around cell {mask_id}")
        [Zm,Ym, Xm] = ndimage.center_of_mass(segmentation_mask == mask_id)
        Y_min = np.max([0, Ym - diam / scale_xy]).astype(int)
        Y_max = np.min([segmentation_mask.shape[1], Ym + diam / scale_xy]).astype(int)
        X_min = np.max([0, Xm - diam / scale_xy]).astype(int)
        X_max = np.min([segmentation_mask.shape[2], Xm + diam / scale_xy]).astype(int)
        crop_mask = mask[:, Y_min:Y_max, X_min:X_max]
        threshold = detection.automated_threshold_setting(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask)

        spots, _ = detection.spots_thresholding(rna_log[:, Y_min:Y_max, X_min:X_max], crop_mask, threshold)

        if min_cos_tetha is not None:
            gz, gy, gx = np.gradient(rna_gaus[:,Y_min:Y_max, X_min:X_max])
            new_spots = []
            for s in spots:
                if mean_cos_tetha(gy, gx, z=s[0], yc=s[1], xc=s[2], order=order) > min_cos_tetha:
                    new_spots.append(s)

        if test_mode: ## test mode
            input = np.amax(rna[:,Y_min:Y_max,  X_min:X_max], 0)
            pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
            rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
            fig, ax = plt.subplots(2, 1, figsize=(15, 15))
            plt.title(f' X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}', fontsize=20)

            ax[0].imshow(rna_scale)
            ax[1].imshow(rna_scale)
            for s in spots:
                ax[0].scatter(s[-1], s[-2], c='red', s=28)
            plt.show()

            fig, ax = plt.subplots(2, 1, figsize=(15, 15))
            plt.title(f'with remove artf  order{order}  min_tetha {min_cos_tetha}  X {str([X_min, X_max])} + Y {str([Y_min, Y_max])}', fontsize=20)
            ax[0].imshow(rna_scale)
            ax[1].imshow(rna_scale)
            for s in new_spots:
                ax[0].scatter(s[-1], s[-2], c='red', s=28)
            plt.show()
        spots = new_spots
        spots = np.array(spots)
        if len(spots) > 0:
            spots = spots + np.array([0, Y_min, X_min])
            all_spots += list(spots)

    all_spots = remove_double_detection(input_array = np.array(all_spots),
                threshold =threshold_merge_limit,
                scale_z_xy = np.array([0.300, 0.103, 0.103]))

    if test_mode:
        input = np.amax(rna, 0)
        pa_ch1, pb_ch1 = np.percentile(input, (1, 99))
        rna_scale = rescale_intensity(input, in_range=(pa_ch1, pb_ch1), out_range=np.uint8).astype('uint8')
        fig, ax = plt.subplots(2, 1, figsize=(40, 40))
        ax[0].imshow(rna_scale)
        ax[1].imshow(rna_scale)
        for s in all_spots:
            ax[0].scatter(s[-1], s[-2], c='red', s=28)
        plt.show()

    return all_spots
































