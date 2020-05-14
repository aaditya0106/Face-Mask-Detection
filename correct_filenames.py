# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:45:38 2020

@author: HP
"""


from os import listdir,rename

train_img_without_mask = "Dataset/train/without_mask"
train_img_with_mask = "Dataset/train/with_mask"
test_img_without_mask = "Dataset/test/without_mask"
test_img_with_mask = "Dataset/test/with_mask"

for count, filename in enumerate(listdir(train_img_with_mask)): 
    dst = train_img_with_mask + "/with_mask_" + str(count) + ".jpg"
    src = train_img_with_mask + "/" + filename
    rename(src, dst) 
    
for count, filename in enumerate(listdir(train_img_without_mask)): 
    dst = train_img_without_mask + "/without_mask_" + str(count) + ".jpg"
    src = train_img_without_mask + "/" + filename
    rename(src, dst)
    
for count, filename in enumerate(listdir(test_img_with_mask)): 
    dst = test_img_with_mask + "/with_mask_" + str(count) + ".jpg"
    src = test_img_with_mask + "/" + filename
    rename(src, dst) 
    
for count, filename in enumerate(listdir(test_img_without_mask)): 
    dst = test_img_without_mask + "/without_mask_" + str(count) + ".jpg"
    src = test_img_without_mask + "/" + filename
    rename(src, dst)
