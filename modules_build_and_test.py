# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

def estimate_bounding_box(poseKeypoints,
                          img_width,
                          img_height,
                          ratio_x,
                          ratio_y):
    #
    # input:
    #   poseKeypoints - tuple of key points from openpose
    #   img_width - width of input image
    #   img_height - height of input image
    #   ratio_x - percentage of bounding box's width increase
    #   ratio_y - percentage of bounding box's height increase
    # output:
    #   list of bounding boxes estimated
    #

    # each bounding box - x1,y1,x2,y2
    #   where (x1,y1) is top-left corner point coordinate
    #         (x2,y2) is right-bottom corner point coordinate
    bxs = ()

    # check validity
    if poseKeypoints.size < 2:
        return []

    num_objs = len(poseKeypoints)
    for i in range(0, num_objs):
        obj = poseKeypoints[i]
        Xs = [item for item in obj[:,0] if item > 0]
        Ys = [item for item in obj[:,1] if item > 0]

        x1 = np.min(Xs)
        x2 = np.max(Xs)
        y1 = np.min(Ys)
        y2 = np.max(Ys)

        delta_width = (x2-x1)*ratio_x/2
        delta_height = (y2-y1)*ratio_y/2

        x1_new = (x1 - delta_width) if (x1 - delta_width) > 0 else 0
        y1_new = (y1 - delta_height) if (y1 - delta_height) > 0 else 0
        x2_new = (x2 + delta_width) if (x2 + delta_width) < img_width else img_width
        y2_new = (y2 + delta_height) if (y2 + delta_height) < img_height else img_height

        bx = (int(x1_new), int(y1_new), int(x2_new), int(y2_new)),
        bxs += bx

    return bxs

def extract_ROI_and_HOG_feature(datum, roi_and_hog_path):

    img_height, img_width, _ = datum.cvOutputData.shape
    # estimate bounding boxes
    bxs = estimate_bounding_box(datum.poseKeypoints,
                                img_width,
                                img_height,
                                0.10,
                                0.10)

    # draw bounding boxes on output image
    bnb_img = datum.cvOutputData
    for bx in bxs:
        cv2.rectangle(bnb_img,
                      (bx[0], bx[1]),
                      (bx[2], bx[3]),
                      color=(255, 0, 0),
                      thickness=1)

    # display image with bounding boxes
    # resized_bnb_img = cv2.resize(bnb_img, (960, 540))
    # cv2.imshow("Bounding box cropped", resized_bnb_img)
    # cv2.waitKey(0)

    # save region of interest and HOG feature array to file
    for i in range(0, len(bxs)):
        roi_img = datum.cvInputData[bxs[i][1]:bxs[i][3], bxs[i][0]:bxs[i][2]]
        # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", roi_img)
        # cv2.waitKey(0)

        # resize roi image to 64x64
        winSize = (64, 64)
        resized_roi_img = cv2.resize(roi_img, winSize)

        # save roi_img to file
        roi_img_path = roi_and_hog_path + img_name + "_roi_" + str(i) + ".png"
        cv2.imwrite(roi_img_path, resized_roi_img)

        # initialize HOG descriptor
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

        # compute HOG features for resized ROI image
        hog_desc = hog.compute(resized_roi_img)

        # reshape HOG feature as gray-scale image
        hog_desc_image = np.array(hog_desc.reshape(42, 42)) * 255

        # save hog descriptor to file as a gray-scale image
        hog_descriptor_path = roi_and_hog_path + img_name + "_hog_" + str(i) + ".png"
        cv2.imwrite(hog_descriptor_path, hog_desc_image)

def calc_means_stds_of_keypoints_set(set_of_keypoints):
    # calculate mean and std values of a keypoints set

    Xs = [item for item in set_of_keypoints[:, 0] if item > 0]
    Ys = [item for item in set_of_keypoints[:, 1] if item > 0]
    # mean and std of Xs
    meanX = np.mean(Xs)
    stdX = np.std(Xs)
    # mean and std of Ys
    meanY = np.mean(Ys)
    stdY = np.std(Ys)

    return ((meanX, meanY),(stdX, stdY))

def bounding_box_merging(poseKeypoints,
                         ratio_of_intersec_thresh,
                         ratio_of_distance_thresh
                         ):
    # inputs:
    #   - poseKeypoints - list of sets of pose key points of an image
    #   - ratio_of_intersec_thresh - no. intersection points/ min no. 2 set of indexed keypoints
    #     we will merge the 2 sets if this ratio less than appropriate input param
    #   - ratio_of_distance_thresh - (delta mean)/(sum of std of 2 sets)
    #     we will merge the 2 sets if this ratio less than appropriate input param
    # outputs:
    #   - array of merged sets of keypoints

    print("Bounding box merging module.")
    list_of_sets_of_keypoints_output = list()

    # calculate mean and std values of each keypoints set
    # check validity
    if poseKeypoints.size < 2:
        return []

    num_objs = len(poseKeypoints)
    list_of_sets_of_keypoints = list()
    # list of tuple of (mean,std) of each keypoints set
    list_of_means_stds = list()
    for i in range(0, num_objs):
        obj = poseKeypoints[i]
        list_of_sets_of_keypoints.append(obj)

        # calculate means and stds
        obj_stat = calc_means_stds_of_keypoints_set(obj)
        list_of_means_stds.append(obj_stat)

        # print("keyjoints set:" + str(obj) + "\n")
        # print("mean and std: " + str(obj_stat) + "\n")

    # do analysis for merging
    while len(list_of_sets_of_keypoints) > 0:
        # denote each set of poseKeypoints as a candidate
        # thus, list of sets of keypoints as list of candidates

        # if there is only one candidate, finalize the job by
        # adding this candidate to list of output
        if len(list_of_sets_of_keypoints) == 1:
            list_of_sets_of_keypoints_output.append(list_of_sets_of_keypoints.pop(0))
            list_of_means_stds.pop(0)
            break

        # if there are more than one candidate
        # get the first candidate
        fst_candidate = list_of_sets_of_keypoints.pop(0)
        ((fst_meanX, fst_meanY), (fst_stdX, fst_stdY)) = list_of_means_stds.pop(0)

        merged_idxs = []
        for cdt_idx in range(0,len(list_of_sets_of_keypoints)):
            # pick second candidate
            snd_candidate = list_of_sets_of_keypoints[cdt_idx]
            # appropriate means and stds
            ((snd_meanX, snd_meanY), (snd_stdX, snd_stdY)) = list_of_means_stds[cdt_idx]

            # try to merge the first and the second candidate
            fst_keypoints_indexes = [i for i in range(0, len(fst_candidate)) if (fst_candidate[i][0] > 0 or fst_candidate[i][1] > 0)]
            snd_keypoints_indexes = [i for i in range(0, len(snd_candidate)) if (snd_candidate[i][0] > 0 or snd_candidate[i][1] > 0)]

            # calculate ratio of intersection
            intersection_idxs = list(set(fst_keypoints_indexes).intersection(snd_keypoints_indexes))
            ratio_of_intersec = (float)(len(intersection_idxs))/min(len(fst_keypoints_indexes),len(snd_keypoints_indexes))
            if ratio_of_intersec < ratio_of_intersec_thresh:
                # calculate ratio of distance
                ratio_of_dist = (float)(np.abs(snd_meanX-fst_meanX))/(2*(fst_stdX+snd_stdX)) + (np.abs(snd_meanY-fst_meanY))/(2*(fst_stdY+snd_stdY))
                if ratio_of_dist < ratio_of_distance_thresh:
                    # both intersec ratio and distance ratio are satisfied
                    # then we merge the second candidate to the first one
                    for idx in snd_keypoints_indexes:
                        # if idx is intersection index
                        if idx in intersection_idxs:
                            # check and take data points with higher probability
                            if fst_candidate[idx][2] < snd_candidate[idx][2]:
                                fst_candidate[idx] = snd_candidate[idx]
                        else:
                            fst_candidate[idx] = snd_candidate[idx]

                    # re-calculate means and stds of the first candidate
                    ((fst_meanX, fst_meanY), (fst_stdX, fst_stdY)) = calc_means_stds_of_keypoints_set(fst_candidate)
                    # record merged index
                    merged_idxs.append(cdt_idx)

        # add the first candidate to result list
        list_of_sets_of_keypoints_output.append(fst_candidate)
        # update current sets of keypoints by removing merged candidates
        list_of_sets_of_keypoints = [list_of_sets_of_keypoints[i] for i in range(0,len(list_of_sets_of_keypoints)) if i not in merged_idxs]
        # update current sets of means and stds appropriately
        list_of_means_stds = [list_of_means_stds[i] for i in range(0,len(list_of_means_stds)) if i not in merged_idxs]

    # ending the merging
    # convert list to ndarray
    arr_of_keypoints_output = np.array(list_of_sets_of_keypoints_output)

    # replace unconfident points from big sets
    arr_of_keypoints_output = replace_unconfident_points_in_big_set_by_more_confident_small_sets(
                                                             poseKeypoints= arr_of_keypoints_output,
                                                             small_set_points_thresh= 4,
                                                             lse_thresh= 1.5,
                                                             confidence_thresh= 0.5,
                                                             unconfidence_thresh= 0.2,
                                                             removed_set_points_thresh= 3
                                                        )

    # print out the result
    print ("The list of %d merged sets of keypoints:\n" % len(arr_of_keypoints_output))
    print(arr_of_keypoints_output)

    return arr_of_keypoints_output

def replace_unconfident_points_in_big_set_by_more_confident_small_sets(poseKeypoints,
                                              small_set_points_thresh,
                                              lse_thresh,
                                              confidence_thresh,
                                              unconfidence_thresh,
                                              removed_set_points_thresh
                                              ):
    #
    # remove duplicated small sets of pose key points
    #
    # inputs:
    #   - poseKeypoints - array of pose keypoints after merging
    #   - small_set_points_thresh - a threshold determine whether a set is small or not
    #   - lse_thresh - least squared error threshold determine whether we consider
    #     a small set (A) as replacement of a subset of points (B) in a given  big set
    #     let's say, A and B are not too far from each other
    #   - confidence_thresh - the minimum avg of prob of small set we will consider
    #   - unconfidence_thresh - the maximum avg of prob of big set we will consider
    #   - removed_set_points_thresh - a threshold determine whether a set will be removed
    #     after all replacement jobs are done
    # output:
    #   - array of pose keypoints after removing duplicates
    #

    # check validity
    if poseKeypoints.size < 2:
        return []

    # build small sets indexes and the rest indexes
    small_sets_idxs = [idx for idx in range(0,len(poseKeypoints)) if np.sum(((poseKeypoints[idx][:,0]>0) | (poseKeypoints[idx][:,1]>0)) == True) <= small_set_points_thresh]
    big_sets_idxs = [idx for idx in range(0,len(poseKeypoints)) if idx not in small_sets_idxs]

    # the indexes of small sets are used for replacement
    replaced_small_set_idxs = []
    for small_set_idx in small_sets_idxs:
        # get small set instance
        sml_inst = poseKeypoints[small_set_idx]
        sml_keypoints_idxs = [keyp_idx for keyp_idx in range(0,len(sml_inst)) if (sml_inst[keyp_idx,0]>0 or sml_inst[keyp_idx,1]>0)]

        for big_set_idx in big_sets_idxs:
            # get big set instance
            big_inst = poseKeypoints[big_set_idx]

            # calculate least squared error of small set compared to appropriate elements in big set
            # ignore the calc if an element of the big one doesn't exist data at this index
            lse = 0.0
            count = 0
            for sml_kp_idx in sml_keypoints_idxs:
                big_kp_X = big_inst[sml_kp_idx, 0]
                big_kp_Y = big_inst[sml_kp_idx, 1]
                if big_kp_X > 0 or big_kp_Y > 0:
                    sml_kp_X = sml_inst[sml_kp_idx, 0]
                    sml_kp_Y = sml_inst[sml_kp_idx, 1]

                    # normalize the value by dividing by the std of the small set
                    ((_,_), (std_sml_X, std_sml_Y)) =  calc_means_stds_of_keypoints_set(sml_inst)
                    lse = lse + np.sqrt((sml_kp_X-big_kp_X)**2 + (sml_kp_Y-big_kp_Y)**2)/(std_sml_X+std_sml_Y)
                    count = count + 1

            if lse > 0:
                lse = lse/count
                # check if we can replace by the current small set
                if lse < lse_thresh:
                    # these two sets close enough to consider
                    # calculate the average probability of small set and big set
                    sml_set_prob_avg = np.average([pct for pct in sml_inst[sml_keypoints_idxs, 2]])
                    big_set_prob_avg = np.average([pct for pct in big_inst[sml_keypoints_idxs, 2]])

                    # check if they can be replaced
                    if sml_set_prob_avg > confidence_thresh and big_set_prob_avg < unconfidence_thresh:
                        # do the replacement
                        for sml_kp_idx in sml_keypoints_idxs:
                            big_inst[sml_kp_idx] = sml_inst[sml_kp_idx]
                        poseKeypoints[big_set_idx] = big_inst

                        # record the replaced index
                        replaced_small_set_idxs.append(small_set_idx)
                        break

    # indexes of sets of keypoints we can remove
    removed_set_points_idxs = [idx for idx in range(0,len(poseKeypoints)) if np.sum(((poseKeypoints[idx][:,0]>0) | (poseKeypoints[idx][:,1]>0)) == True) <= removed_set_points_thresh]

    # union removed sets indexes with replaced sets indexes
    combined_set_points_idxs = list(set(removed_set_points_idxs).union(set(replaced_small_set_idxs)))

    # rebuild the array of keypoints after removing duplicated small sets
    arr_of_keypoints_set_output = np.delete(poseKeypoints, combined_set_points_idxs, axis=0)

    return arr_of_keypoints_set_output

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/dependencies/openpose/libs');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/dependencies/openpose/dlls;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", default="yoga-pose-dataset/train/warrior2/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--image_dir", default="media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./dependencies/openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    image_dir = args[0].image_dir
    imagePaths = op.get_images_on_directory(image_dir)
    # start = time.time()

    # create ROI and HOG directory
    roi_and_hog_path = image_dir + "RoI-HOG/"
    if not os.path.exists(roi_and_hog_path):
        os.mkdir(image_dir + "RoI-HOG/")

    # Process and display images
    for imagePath in imagePaths:

        print("\nCurrent image: " + imagePath + "\n")

        # get image file name
        img_name = imagePath[imagePath.rindex('\\')+1:imagePath.rindex('.')]

        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)

        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        print("Body keypoints: \n" + str(datum.poseKeypoints))

        # extract regions of interest and HOG features
        # extract_ROI_and_HOG_feature(datum, roi_and_hog_path)

        bounding_box_merging(datum.poseKeypoints,0.36,2)

        # estimate boundinig boxes
        img_height, img_width, _ = datum.cvOutputData.shape
        bxs = estimate_bounding_box(datum.poseKeypoints,
                                    img_width,
                                    img_height,
                                    0.10,
                                    0.10)

        # draw bounding boxes on output image
        bnb_img = datum.cvOutputData
        i = 1
        for bx in bxs:
            cv2.rectangle(bnb_img,
                          (bx[0], bx[1]),
                          (bx[2], bx[3]),
                          color=(255, 0, 0),
                          thickness=1)
            cv2.putText(bnb_img,
            "Box " + str(i),
            org=(bx[0], bx[1] - 10),
            color=(0, 0, 250),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            lineType=cv2.LINE_AA,
            thickness=1)
            i = i + 1

        # display image with bounding boxes
        resized_bnb_img = cv2.resize(bnb_img, (960, 540))
        cv2.imshow("Bounding box cropped", resized_bnb_img)
        cv2.waitKey(0)

        # if not args[0].no_display:
        #     resized_img = cv2.resize(datum.cvOutputData, (960,540))
        #     cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", resized_img)
        #     key = cv2.waitKey(0)

        if key == 27: break

    # end = time.time()
    # print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
