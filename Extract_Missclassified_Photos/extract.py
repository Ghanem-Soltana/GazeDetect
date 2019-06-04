# coding=utf8
import sys


sys.path.insert(0, '../BuildTrainingSet')
sys.path.insert(1, '../learning')
import lableImagesByCategory as ruler
import DNN_See_Config as DNN
import configparser
from glob import glob
import os
from termcolor import colored
import ntpath
import alexnet as model2
import torch
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment
import openpyxl.cell
import json

def extractThreePoints(val):
    return val.replace("(","").replace(")","").split(", ")

def convertPointsToList(json_obj, key):
    raw = json_obj[key]
    return flattenVector(raw)

def  flattenVector(raw):
    flattend = list()
    if(isinstance(raw, str)):
        trois_pts = extractThreePoints(raw)
        for inner in trois_pts:
            flattend.append(inner)
        return  flattend
    else:
        for out in raw:
            trois_pts = extractThreePoints(out)
            for inner in trois_pts:
                flattend.append(inner)
        return flattend


def getFileName():
    fileName = 'output.xlsx'
    nb =0
    exists = os.path.isfile(fileName)
    while exists:
        nb = nb+1
        fileName = 'output'+str(nb)+'.xlsx'
        exists = os.path.isfile(fileName)
    return  fileName

def loadDNN (modelLocation):
        net = model2.AlexNet()
        weights = torch.load(modelLocation)
        net.load_state_dict(weights)
        net.eval()
        return net, weights

def convertProba(output):
    classes = ['TopRight', 'BottomCenter', 'BottomLeft', 'BottomRight', 'MiddleLeft', 'MiddleRight', 'TopCenter',
               'TopLeft']

    res = ""
    j=0
    for i in classes:
        res = res + str(i) + " : " + str(output[0][j]) + " ; "
        j+=1

    return res

if __name__ == '__main__':
    # read vars
    config = configparser.ConfigParser()
    config.read("config.ini")

    # Put path the location of images to classify
    images = ruler.readStringVar(config, "myvars", "initialImagesWithJson")
    relativePathDir = ruler.readStringVar(config, "myvars", "relativePathToNNDir")
    modelLocation = ruler.readStringVar(config, "myvars", "modelLocation")

    print ("Extraction of misclassified photos")
    json_fns = glob(images + "/*.json")
    nb_images = len(json_fns)
    nb_missClassified = 0
    net, weights = loadDNN(modelLocation)
    df = pd.DataFrame(columns=["image",
                               "expected_class",
                               "predicted_class",
                               "link_to_image",
                               "interior_margin_2d_0",
                               "interior_margin_2d_1",
                               "interior_margin_2d_2",
    "interior_margin_2d_3",
    "interior_margin_2d_4",
    "interior_margin_2d_5",
    "interior_margin_2d_6",
    "interior_margin_2d_7",
    "interior_margin_2d_8",
    "interior_margin_2d_9",
    "interior_margin_2d_10",
    "interior_margin_2d_11",
    "interior_margin_2d_12",
    "interior_margin_2d_13",
    "interior_margin_2d_14",
    "interior_margin_2d_15",
    "interior_margin_2d_16",
    "interior_margin_2d_17",
    "interior_margin_2d_18",
    "interior_margin_2d_19",
    "interior_margin_2d_20",
    "interior_margin_2d_21",
    "interior_margin_2d_22",
    "interior_margin_2d_23",
    "interior_margin_2d_24",
    "interior_margin_2d_25",
    "interior_margin_2d_26",
    "interior_margin_2d_27",
    "interior_margin_2d_28",
    "interior_margin_2d_29",
    "interior_margin_2d_30",
    "interior_margin_2d_31",
    "interior_margin_2d_32",
    "interior_margin_2d_33",
    "interior_margin_2d_34",
    "interior_margin_2d_35",
    "interior_margin_2d_36",
    "interior_margin_2d_37",
    "interior_margin_2d_38",
    "interior_margin_2d_39",
    "interior_margin_2d_40",
    "interior_margin_2d_41",
    "interior_margin_2d_42",
    "interior_margin_2d_43",
    "interior_margin_2d_44",
    "interior_margin_2d_45",
    "interior_margin_2d_46",
    "interior_margin_2d_47",
    "caruncle_2d_0",
    "caruncle_2d_1",
    "caruncle_2d_2",
    "caruncle_2d_3",
    "caruncle_2d_4",
    "caruncle_2d_5",
    "caruncle_2d_6",
    "caruncle_2d_7",
    "caruncle_2d_8",
    "caruncle_2d_9",
    "caruncle_2d_10",
    "caruncle_2d_11",
    "caruncle_2d_12",
    "caruncle_2d_13",
    "caruncle_2d_14",
    "caruncle_2d_15",
    "caruncle_2d_16",
    "caruncle_2d_17",
    "caruncle_2d_18",
    "caruncle_2d_19",
    "caruncle_2d_20",
    "iris_2d_0",
    "iris_2d_1",
    "iris_2d_2",
    "iris_2d_3",
    "iris_2d_4",
    "iris_2d_5",
    "iris_2d_6",
    "iris_2d_7",
    "iris_2d_8",
    "iris_2d_9",
    "iris_2d_10",
    "iris_2d_11",
    "iris_2d_12",
    "iris_2d_13",
    "iris_2d_14",
    "iris_2d_15",
    "iris_2d_16",
    "iris_2d_17",
    "iris_2d_18",
    "iris_2d_19",
    "iris_2d_20",
    "iris_2d_21",
    "iris_2d_22",
    "iris_2d_23",
    "iris_2d_24",
    "iris_2d_25",
    "iris_2d_26",
    "iris_2d_27",
    "iris_2d_28",
    "iris_2d_29",
    "iris_2d_30",
    "iris_2d_31",
    "iris_2d_32",
    "head_pose_0",
    "head_pose_1",
    "head_pose_2",
    "look_vec_0",
    "look_vec_1",
    "look_vec_2",
    "look_vec_3",
    "pupil_size",
    "iris_size",
    "iris_texture",
    "light_rotation_0",
    "light_rotation_1",
    "light_rotation_2",
    "skybox_texture",
    "skybox_exposure",
    "skybox_rotation",
    "ambient_intensity",
    "light_intensity",
    "pca_shape_coeffs_0",
    "pca_shape_coeffs_1",
    "pca_shape_coeffs_2",
    "pca_shape_coeffs_3",
    "pca_shape_coeffs_4",
    "pca_shape_coeffs_5",
    "pca_shape_coeffs_6",
    "pca_shape_coeffs_7",
    "pca_shape_coeffs_8",
    "pca_shape_coeffs_9",
    "pca_shape_coeffs_10",
    "pca_shape_coeffs_11",
    "pca_shape_coeffs_12",
    "pca_shape_coeffs_13",
    "pca_shape_coeffs_14",
    "pca_shape_coeffs_15",
    "pca_shape_coeffs_16",
    "pca_shape_coeffs_17",
    "pca_shape_coeffs_18",
    "primary_skin_texture"
                               ])

    for json_fn in json_fns:
        print(json_fn)
        # check that we have everything we need: image +json file
        exists = os.path.isfile("%s.jpg" % json_fn[:-5])
        if not exists:
            print("Image does not exist for the json file: " + json_fn)
            nb_images = nb_images -1
            continue

        # guess what the class of the image should be
        imageName = ntpath.basename(json_fn).replace(".json", ".jpg")
        expected = ruler.getLabelFromSinglePhoto(json_fn)
        print(colored("Expected label for "+imageName+"==>"+expected, 'red'))
        #now, we know what is the initial class of the image
        #predict what is the class of the image using the NN we have already trained
        predicted,output = DNN.classifyOneImage(imageName, net, weights,-1,relativePathDir,images)
        print(colored("Class returned by the DNN for "+imageName+"==>"+predicted, 'green'))


        #missclassified
        if(expected.strip().lower()!=predicted.strip().lower()):
            nb_missClassified = nb_missClassified + 1
            json_fn=json_fn.replace("/","\\")
            print(json_fn)
            #stext = open(json_fn, 'r').read()
            f = open(json_fn, 'r')
            json_obj = json.load(f)
            print(type(json_obj))

            interior_margin_2d= convertPointsToList(json_obj, "interior_margin_2d")
            caruncle_2d= convertPointsToList(json_obj, "caruncle_2d")
            iris_2d = convertPointsToList(json_obj, "iris_2d")
            head_pose = convertPointsToList(json_obj, "head_pose")
            eye_details = json_obj["eye_details"]
            look_vec = flattenVector(eye_details["look_vec"])
            pupil_size = eye_details["pupil_size"]
            iris_size = eye_details["iris_size"]
            iris_texture = eye_details["iris_texture"]


            lighting_details = json_obj["lighting_details"]
            light_rotation = flattenVector(lighting_details["light_rotation"])
            skybox_texture = lighting_details["skybox_texture"]
            skybox_exposure = lighting_details["skybox_exposure"]
            skybox_rotation = lighting_details["skybox_rotation"]
            ambient_intensity = lighting_details["ambient_intensity"]
            light_intensity = lighting_details["light_intensity"]

            eye_region_details = json_obj["eye_region_details"]
            pca_shape_coeffs = flattenVector(eye_region_details["pca_shape_coeffs"])
            primary_skin_texture = eye_region_details["primary_skin_texture"]

            df = df.append({"image":imageName,
                            "expected_class":expected,
                            "predicted_class":predicted,
                            "link_to_image":'=HYPERLINK("' + json_fn.replace(".json", ".jpg") + '")',
                            "interior_margin_2d_0": interior_margin_2d[0],
                            "interior_margin_2d_1": interior_margin_2d[1],
                            "interior_margin_2d_2": interior_margin_2d[2],
                            "interior_margin_2d_3": interior_margin_2d[3],
                            "interior_margin_2d_4": interior_margin_2d[4],
                            "interior_margin_2d_5": interior_margin_2d[5],
                            "interior_margin_2d_6": interior_margin_2d[6],
                            "interior_margin_2d_7": interior_margin_2d[7],
                            "interior_margin_2d_8": interior_margin_2d[8],
                            "interior_margin_2d_9": interior_margin_2d[9],
                            "interior_margin_2d_10": interior_margin_2d[10],
                            "interior_margin_2d_11": interior_margin_2d[11],
                            "interior_margin_2d_12": interior_margin_2d[12],
                            "interior_margin_2d_13": interior_margin_2d[13],
                            "interior_margin_2d_14": interior_margin_2d[14],
                            "interior_margin_2d_15": interior_margin_2d[15],
                            "interior_margin_2d_16": interior_margin_2d[16],
                            "interior_margin_2d_17": interior_margin_2d[17],
                            "interior_margin_2d_18": interior_margin_2d[18],
                            "interior_margin_2d_19": interior_margin_2d[19],
                            "interior_margin_2d_20": interior_margin_2d[20],
                            "interior_margin_2d_21": interior_margin_2d[21],
                            "interior_margin_2d_22": interior_margin_2d[22],
                            "interior_margin_2d_23": interior_margin_2d[23],
                            "interior_margin_2d_24": interior_margin_2d[24],
                            "interior_margin_2d_25": interior_margin_2d[25],
                            "interior_margin_2d_26": interior_margin_2d[26],
                            "interior_margin_2d_27": interior_margin_2d[27],
                            "interior_margin_2d_28": interior_margin_2d[28],
                            "interior_margin_2d_29": interior_margin_2d[29],
                            "interior_margin_2d_30": interior_margin_2d[30],
                            "interior_margin_2d_31": interior_margin_2d[31],
                            "interior_margin_2d_32": interior_margin_2d[32],
                            "interior_margin_2d_33": interior_margin_2d[33],
                            "interior_margin_2d_34": interior_margin_2d[34],
                            "interior_margin_2d_35": interior_margin_2d[35],
                            "interior_margin_2d_36": interior_margin_2d[36],
                            "interior_margin_2d_37": interior_margin_2d[37],
                            "interior_margin_2d_38": interior_margin_2d[38],
                            "interior_margin_2d_39": interior_margin_2d[39],
                            "interior_margin_2d_40": interior_margin_2d[40],
                            "interior_margin_2d_41": interior_margin_2d[41],
                            "interior_margin_2d_42": interior_margin_2d[42],
                            "interior_margin_2d_43": interior_margin_2d[43],
                            "interior_margin_2d_44": interior_margin_2d[44],
                            "interior_margin_2d_45": interior_margin_2d[45],
                            "interior_margin_2d_46": interior_margin_2d[46],
                            "interior_margin_2d_47": interior_margin_2d[47],
                            "caruncle_2d_0": caruncle_2d[0],
                            "caruncle_2d_1": caruncle_2d[1],
                            "caruncle_2d_2": caruncle_2d[2],
                            "caruncle_2d_3": caruncle_2d[3],
                            "caruncle_2d_4": caruncle_2d[4],
                            "caruncle_2d_5": caruncle_2d[5],
                            "caruncle_2d_6": caruncle_2d[6],
                            "caruncle_2d_7": caruncle_2d[7],
                            "caruncle_2d_8": caruncle_2d[8],
                            "caruncle_2d_9": caruncle_2d[9],
                            "caruncle_2d_10": caruncle_2d[10],
                            "caruncle_2d_11": caruncle_2d[11],
                            "caruncle_2d_12": caruncle_2d[12],
                            "caruncle_2d_13": caruncle_2d[13],
                            "caruncle_2d_14": caruncle_2d[14],
                            "caruncle_2d_15": caruncle_2d[15],
                            "caruncle_2d_16": caruncle_2d[16],
                            "caruncle_2d_17": caruncle_2d[17],
                            "caruncle_2d_18": caruncle_2d[18],
                            "caruncle_2d_19": caruncle_2d[19],
                            "caruncle_2d_20": caruncle_2d[20],
                            "iris_2d_0": iris_2d[0],
                            "iris_2d_1": iris_2d[1],
                            "iris_2d_2": iris_2d[2],
                            "iris_2d_3": iris_2d[3],
                            "iris_2d_4": iris_2d[4],
                            "iris_2d_5": iris_2d[5],
                            "iris_2d_6": iris_2d[6],
                            "iris_2d_7": iris_2d[7],
                            "iris_2d_8": iris_2d[8],
                            "iris_2d_9": iris_2d[9],
                            "iris_2d_10": iris_2d[10],
                            "iris_2d_11": iris_2d[11],
                            "iris_2d_12": iris_2d[12],
                            "iris_2d_13": iris_2d[13],
                            "iris_2d_14": iris_2d[14],
                            "iris_2d_15": iris_2d[15],
                            "iris_2d_16": iris_2d[16],
                            "iris_2d_17": iris_2d[17],
                            "iris_2d_18": iris_2d[18],
                            "iris_2d_19": iris_2d[19],
                            "iris_2d_20": iris_2d[20],
                            "iris_2d_21": iris_2d[21],
                            "iris_2d_22": iris_2d[22],
                            "iris_2d_23": iris_2d[23],
                            "iris_2d_24": iris_2d[24],
                            "iris_2d_25": iris_2d[25],
                            "iris_2d_26": iris_2d[26],
                            "iris_2d_27": iris_2d[27],
                            "iris_2d_28": iris_2d[28],
                            "iris_2d_29": iris_2d[29],
                            "iris_2d_30": iris_2d[30],
                            "iris_2d_31": iris_2d[31],
                            "iris_2d_32": iris_2d[32],
                            "head_pose_0": head_pose[0],
                            "head_pose_1": head_pose[1],
                            "head_pose_2": head_pose[2],
                            "look_vec_0": look_vec[0],
                            "look_vec_1": look_vec[1],
                            "look_vec_2": look_vec[2],
                            "look_vec_3": look_vec[3],
                            "pupil_size": pupil_size,
                            "iris_size": iris_size,
                            "iris_texture": iris_texture,
                            "light_rotation_0": light_rotation[0],
                            "light_rotation_1": light_rotation[1],
                            "light_rotation_2": light_rotation[2],
                            "skybox_texture": skybox_texture,
                            "skybox_exposure": skybox_exposure,
                            "skybox_rotation": skybox_rotation,
                            "ambient_intensity": ambient_intensity,
                            "light_intensity": light_intensity,
                            "pca_shape_coeffs_0": pca_shape_coeffs[0],
                            "pca_shape_coeffs_1": pca_shape_coeffs[1],
                            "pca_shape_coeffs_2": pca_shape_coeffs[2],
                            "pca_shape_coeffs_3": pca_shape_coeffs[3],
                            "pca_shape_coeffs_4": pca_shape_coeffs[4],
                            "pca_shape_coeffs_5": pca_shape_coeffs[5],
                            "pca_shape_coeffs_6": pca_shape_coeffs[6],
                            "pca_shape_coeffs_7": pca_shape_coeffs[7],
                            "pca_shape_coeffs_8": pca_shape_coeffs[8],
                            "pca_shape_coeffs_9": pca_shape_coeffs[9],
                            "pca_shape_coeffs_10": pca_shape_coeffs[10],
                            "pca_shape_coeffs_11": pca_shape_coeffs[11],
                            "pca_shape_coeffs_12": pca_shape_coeffs[12],
                            "pca_shape_coeffs_13": pca_shape_coeffs[13],
                            "pca_shape_coeffs_14": pca_shape_coeffs[14],
                            "pca_shape_coeffs_15": pca_shape_coeffs[15],
                            "pca_shape_coeffs_16": pca_shape_coeffs[16],
                            "pca_shape_coeffs_17": pca_shape_coeffs[17],
                            "pca_shape_coeffs_18": pca_shape_coeffs[18],
                            "primary_skin_texture":primary_skin_texture
                            #,"Proba_per_class":convertProba(output)
            }, ignore_index=True)

    # save stuff
    fName = getFileName()
    df.to_excel(fName, index=False)
    wb = openpyxl.load_workbook(fName)
    sheet = wb.active

    for x in range(1,300):
        letter = openpyxl.utils.cell.get_column_letter(x)
        print(letter)
        sheet.column_dimensions[letter].width = 20

        for y in range(2, nb_missClassified+2):
            cell = sheet[letter+str(y)]
            cell.alignment = Alignment(horizontal='left',
                                      vertical='top',
                                      text_rotation=0,
                                      wrap_text=True,
                                      shrink_to_fit=True,
                                      indent=0)
            cell.font = Font(size=12)

    wb.save(fName)
    print ("file "+fName+" was generated")
    print(" {} out of {} images were correctly predicted ".format(nb_images - nb_missClassified,nb_images ))
    if(float(nb_images)!=0):
        print("The average accuracy is: {} %".format(100.0 * (nb_images - nb_missClassified) / (float(nb_images))))
    else:
        print("No input image found!")