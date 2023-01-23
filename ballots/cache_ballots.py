import os, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from PIL import Image
from tqdm import tqdm
import pdb

# The classification algorithm
def find_ellipse(filename, plot):
    img_full = cv.imread(filename, cv.IMREAD_COLOR)

    cropRows = [-img_full.shape[0]//3, -img_full.shape[0]//6]
    cropCols = [-img_full.shape[1]//5, -img_full.shape[1]//15]

    img = img_full[cropRows[0]:cropRows[1],cropCols[0]:cropCols[1],:]


    # Check if image is loaded fine
    if img is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = 255-gray

    rows = gray.shape[0]

    contours,hierarchy = cv.findContours(gray,2,1)
    contours_passed = []
    fit_ellipses = []
    areas = []
    area_constraints = [250,1500]
    for i in contours:
        area = cv.contourArea(i)
        if(area >= area_constraints[0] and area <= area_constraints[1]):
            contours_passed += [i]
            curr_ell = cv.fitEllipse(i)
            fit_ellipses += [curr_ell]
            areas += [area]
    areas = np.array(areas)
    analytic_areas = np.array([ell[1][0]*ell[1][1]*np.pi/4.0 for ell in fit_ellipses])
    if areas.shape[0] == 0:
        vote = -1
    else:
        idx_best = np.argmin( np.abs(analytic_areas - areas)/areas )
        ell_best = fit_ellipses[idx_best]
        decision_boundary = int(img.shape[0]/1.5)
        vote = int(fit_ellipses[idx_best][0][1] >= decision_boundary)
        if plot:
            cv.line(img, (0,decision_boundary), (img.shape[1]-1,decision_boundary), (255,0,0), 2)
            cv.ellipse(img, fit_ellipses[idx_best], (0,255,0), -1)

    if plot:
        img_full[cropRows[0]:cropRows[1],cropCols[0]:cropCols[1],:] = img

        plt.figure(figsize=(15,15))
        plt.imshow(img_full)
        plt.axis('off');

    return vote

if __name__ == "__main__":
    # Process .tif images into .png
    base_path = "/Users/angelopoulos/Code/working/prediction-powered-inference/ballots/raw/A22_BallotImages"
    new_path = "/Users/angelopoulos/Code/working/prediction-powered-inference/ballots/proc/"
    ballot_dirnames = []
    ballot_filenames = []
    os.makedirs(new_path, exist_ok = True)
    counter = 1
    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if '.tif' in filename:
                ballot_dirnames += [dirpath,]
                ballot_filenames += [filename,]

    for i in range(len(ballot_filenames)):
        if not os.path.exists(new_path + str(counter) + ".png"):
            img_full = cv.imread(ballot_dirnames[i] + "/" + ballot_filenames[i], cv.IMREAD_COLOR)
            cv.imwrite(new_path + str(counter) + ".png", img_full)
        counter += 1

    # Read in and clean labels
    cal_label_csv = pd.read_csv('labels.csv')
    cal_labeled_image_filenames = [new_path + cal_label_csv['image'][i].split("/")[3] for i in range(len(cal_label_csv))]
    cal_labels = -np.ones((len(cal_label_csv,)))
    cal_labels[cal_label_csv['choice'] == "Matt Haney"] = 1
    cal_labels[cal_label_csv['choice'] == "David Campos"] = 0

    cal_preds = np.array([find_ellipse(fname, plot=False) for fname in cal_labeled_image_filenames])
    clean_cal_preds = cal_preds[(cal_labels >= 0) & (cal_preds >= 0)]
    clean_cal_labels = cal_labels[(cal_labels >= 0) & (cal_preds >= 0)]

    # Model accuracy
    print(f"Model accuracy: {(clean_cal_preds == clean_cal_labels).astype(float).mean()*100:.2f}%")

    # Collect the filenames of the processed ballots
    base_path = "/Users/angelopoulos/Code/working/prediction-powered-inference/ballots/proc/"
    ballot_filenames = []
    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in filenames:
            if '.png' in filename:
                ballot_filenames += [dirpath + "/" + filename, ]

    # Count the prediction-powered votes
    votes = []
    for i in tqdm(range(len(ballot_filenames))):
        ballot_filename = ballot_filenames[i]
        if ballot_filename in cal_labeled_image_filenames: # Don't count the ones we already labeled
            continue
        else:
            votes += [find_ellipse(ballot_filename, plot=False)]

    votes = np.array(votes)
    clean_votes = votes[votes >= 0]

    np.savez('./ballot-data.npz', Y_labeled=clean_cal_labels, Yhat_labeled=clean_cal_preds, Yhat_unlabeled=clean_votes)
