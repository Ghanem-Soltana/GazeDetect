# coding=utf8
import cv2
import numpy as np
import json
from glob import glob
import math
import shutil
import os
import configparser
import ntpath
import sys

debug = False


def readStringVar(config, varCat,varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        return raw

def readBoolVar(config, varCat,varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        if (raw.strip().lower() == "true"):
            return True
        else:
            return False


def milieu(x1, y1, x2, y2):
 x = (x1 + x2) / 2
 y = (y1 + y2) / 2
 return [x, y]


# https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
 px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
   (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
 py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
   (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
 return [px, py]


def process_json_list(json_list,img):
 ldmks = [eval(s) for s in json_list]
 return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

def getLabelFromSinglePhoto (json_fn):
    img = cv2.imread("%s.jpg" % json_fn[:-5])
    #cv2.imshow("syntheseyes_img", img)
    data_file = open(json_fn)
    data = json.load(data_file)

    ldmks_interior_margin = process_json_list(data['interior_margin_2d'],img)

    ldmks_iris = process_json_list(data['iris_2d'],img)

    intersection = findIntersection(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                                    int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                                    int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]),
                                    int(ldmks_interior_margin[4][0]), int(ldmks_interior_margin[4][1]),
                                    int(ldmks_interior_margin[12][0]), int(ldmks_interior_margin[12][1]))

    milieu_x = milieu(int(ldmks_interior_margin[0][0]), int(ldmks_interior_margin[0][1]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][0]),
                      int(ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)][1]))
    milieu_y = milieu(int(ldmks_interior_margin[4][0]), int(ldmks_interior_margin[4][1]),
                      int(ldmks_interior_margin[12][0]), int(ldmks_interior_margin[12][1]))

    look_vec = list(eval(data['eye_details']['look_vec']))

    # Draw green foreground points and lines
    for ldmk in np.vstack([ldmks_interior_margin[0], ldmks_interior_margin[4], ldmks_interior_margin[12],
                           ldmks_interior_margin[round(len(ldmks_interior_margin) / 2)]]):
        cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0, 255, 0), -1)

    eye_c = np.mean(ldmks_iris[:, :2], axis=0).astype(int)
    # print("begin point")
    # print(tuple(eye_c))
    look_vec[1] = -look_vec[1]
    # print ("end point")
    # print (tuple(eye_c+(np.array(look_vec[:2])*80).astype(int)))
    # origin
    point_A = tuple(eye_c)  # horizon
    point_B = tuple(eye_c + (np.array([40, 0]).astype(int)))
    # where the eye look
    point_C = tuple(eye_c + (np.array(look_vec[:2]) * 80).astype(int))

    # horizon
    cv2.line(img, point_A, point_B, (0, 0, 0), 3)
    cv2.line(img, point_A, point_B, (0, 255, 255), 2)
    # where the eye look
    cv2.line(img, point_A, point_C, (0, 0, 0), 3)
    cv2.line(img, point_A, point_C, (0, 255, 255), 2)

    angle = math.atan2(point_C[0] - point_A[0], point_C[1] - point_A[1]) - math.atan2(point_B[0] - point_A[0],
                                                                                      point_B[1] - point_A[1]);

    cv2.circle(img, (int(intersection[0]), int(intersection[1])), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(milieu_x[0]), int(milieu_x[1])), 2, (0, 255, 0), -1)
    cv2.circle(img, (int(milieu_y[0]), int(milieu_y[1])), 2, (0, 255, 0), -1)

    # dist_intersection = math.sqrt(
    #  (point_A[0] - intersection[0]) * (point_A[0] - intersection[0]) + (point_A[1] - intersection[1]) * (
    #     point_A[1] - intersection[1]))
    dist_x = math.sqrt((point_A[0] - milieu_x[0]) * (point_A[0] - milieu_x[0]) + (point_A[1] - milieu_x[1]) * (
            point_A[1] - milieu_x[1]))
    #  dist_y = math.sqrt((point_A[0] - milieu_y[0]) * (point_A[0] - milieu_y[0]) + (point_A[1] - milieu_y[1]) * (
    #     point_A[1] - milieu_y[1]))

    angle = (angle * 180) / math.pi

    while (angle < 0):
        angle = angle + 360

    print("the angle " + str(angle))
    print("dist_x " + str(dist_x))

    classe = "Error"

    try:
        imageName = ntpath.basename(json_fn).replace(".json", ".jpg")
        if angle >= 0 and angle < 22.5:
            classe = "MiddleLeft"
        if angle > 22.5 and angle < 67.5:
            classe = "TopLeft"
        if angle > 67.5 and angle < 112.5:
            classe = "TopCenter"
        if angle > 112.5 and angle < 157.5:
            classe = "TopRight"
        if angle > 157.5 and angle < 202.5:
            classe = "MiddleRight"
        if angle > 202.5 and angle < 247.5:
            classe = "BottomRight"
        if angle > 247.5 and angle < 292.5:
            classe = "BottomCenter"
        if angle > 292.5 and angle < 337.5:
            classe = "BottomLeft"
        if angle >= 337.5:
            classe = "MiddleLeft"

        # if (angle >= 0 and angle < 22.5 or angle >= 337.5 and angle <= 360) or (angle >= 157.5 and angle < 202.5):
        # chance to be center center!!
        # if dist_x <= 29:
        #  classe = "MiddleCenter"
        print("Classe ==>" + classe)
        print("imageName ==>" + imageName)
        print("json_fn ==>" + json_fn)
    except Exception as e:
        print(str(e))
        sys.exit(0)
    return classe

if __name__ == '__main__':
 # read vars
 config = configparser.ConfigParser()
 config.read("config.ini")

 # Put path the location of images to classify
 images = "images"
 images = readStringVar(config, "myvars", "images")
 json_fns = glob(images + "/*.json")
 classes = ['TopRight', 'BottomCenter', 'BottomLeft', 'BottomRight', 'MiddleLeft', 'MiddleRight', 'TopCenter',
            'TopLeft']

 for c in classes:
     if not os.path.isdir("outputs/" + c):
         try:
             os.mkdir("outputs/" + c)
         except OSError:
             print("Creation of the directory %s failed" % "outputs/" + c)
         else:
             print("Successfully created the directory %s " % "outputs/" + c)


 for json_fn in json_fns:
  print(json_fn)
  
  exists = os.path.isfile("%s.jpg"%json_fn[:-5])
  if not exists:
   print("Image does not exist for the json file: "+json_fn)
   continue
   
  classe = getLabelFromSinglePhoto(json_fn)
  imageName = ntpath.basename(json_fn).replace(".json", ".jpg")
  if(classe !=  "Error"):
   shutil.copyfile(images+"/" + imageName, "outputs/"+classe+"/"+imageName)
  if(debug ==  True):
   cv2.waitKey(200)
   input('Press key ')


