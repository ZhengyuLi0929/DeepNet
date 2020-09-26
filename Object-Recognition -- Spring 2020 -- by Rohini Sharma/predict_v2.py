import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import ResNet152
from tqdm import tqdm
import human_categories as hc
import csv
import os




Categories = hc.HumanCategories()

def predict(img):
    model = ResNet152()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet.preprocess_input(x)
    predictions = model.predict(x)
    all_cat_pred = predictions[0]
    predicted_classes = resnet.decode_predictions(predictions, top=9)
    return predictions[0]

    #return predicted_classes

def get_cats_from_preds(all_preds):
    human_cats = hc.get_human_object_recognition_categories()
    human_cat_prbs = {i : 0.0 for i in human_cats}
    cats = open("categories.txt")
    for i,line in enumerate(cats):
        WNID = line.split(" ")[0]
        human_WNID_cat = Categories.get_human_category_from_WNID(WNID)
        if human_WNID_cat in  human_cats:
            current = human_cat_prbs[human_WNID_cat]
            human_cat_prbs[human_WNID_cat] = all_preds[i] + current
    return human_cat_prbs


def get_max_prob_cat(preds_map):
    max = 0
    maxCat = None
    for cat,prob in preds_map.items():
        if prob > max:
            max = prob
            maxCat = cat
    return maxCat, max

def get_human_cateogory(predictions):

    best_probability = -1
    category = None
    for imagenet_id, name, likelihood in predictions[0]:
        if likelihood > best_probability:
            best_probability = likelihood
            category = Categories.get_human_category_from_WNID(imagenet_id)
            if category == None:
                category = name
                print(" - {}: {:2f} likelihood".format(name, likelihood))
    return category

def write_to_csv(actual, prediction_map, image_num):
    actual = "Actual: " + actual
    row_to_write = [[image_num, actual, prediction_map]]
    with open ("resnet150_unsegmented.csv",'a') as file:
        writer = csv.writer(file)
        writer.writerows(row_to_write)



def sort_map(to_sort):
    to_return = []
    sorted_cat_list = sorted(to_sort, key=to_sort.__getitem__)
    for cat in sorted_cat_list:
        to_return.append(cat + ":" + str(to_sort[cat]))

    return list(reversed(to_return))



def classify_imgs_in_dir(directory, real_cat):
    num = 0
    print(directory)
    for im in tqdm(os.listdir(directory)):
        if im.split("_")[0] == ".DS":
            continue
        img_obj = image.load_img(directory+im, target_size = (224,224))
        predictions = predict(img_obj)
        pred_map = get_cats_from_preds(predictions)
        sorted_map = sort_map(pred_map)

        cat, prob = get_max_prob_cat(pred_map)
        write_to_csv(real_cat, sorted_map, im)
        num += 1
    print(num)
#category_names = ["Knife", "Keyboard", "Elephant", "Bicycle", "Airplane",
 #           "Clock", "Oven", "Chair", "Bear", "Boat",
  #         "Car", "Bird", "Dog", "Orange", "Refrigerator", "Bowl"]

def classify_all_imgs():
    category_names = ["Knife", "Keyboard", "Elephant", "Bicycle", "Airplane",
            "Clock", "Oven", "Chair", "Bear", "Boat",
           "Car", "Bird", "Dog", "Orange", "Refrigerator", "Bowl"]
    for cat in category_names:
        path = "/Users/rohinisharma/Projects/MURIproj/" + cat + "/"
        classify_imgs_in_dir(path, cat)
        #path = "/Users/rohinisharma/Projects/MURIproj/" + cat + "_Without_Background/"
        #classify_imgs_in_dir(path, cat)

classify_all_imgs()



