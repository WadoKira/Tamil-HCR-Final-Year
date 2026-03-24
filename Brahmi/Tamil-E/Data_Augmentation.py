import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os

original_image_base_path = "C:/Users/HP/Pictures/Screenshots/"

train_base_dir = "E:/Capstone/Capstone_Tamizhi/archive/train/"
test_base_dir = "E:/Capstone/Capstone_Tamizhi/archive/test/"

for num in range(160, 161):
    image_path = f"{original_image_base_path}{num}.png"
    image = load_img(image_path, color_mode="grayscale")
    image_array = img_to_array(image)  
    image_array = np.expand_dims(image_array, axis=0)  

    datagen = ImageDataGenerator(
        #rotation_range=20,         
        width_shift_range=0.1,     
        height_shift_range=0.1,    
        zoom_range=0.1,            
        shear_range=0.1,           
        brightness_range=[0.8, 1.2] 
    )

    train_save_dir = os.path.join(train_base_dir, str(num))
    test_save_dir = os.path.join(test_base_dir, str(num))
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    num_train_images = 25
    i = 0
    for batch in datagen.flow(image_array, batch_size=1, save_to_dir=train_save_dir, save_prefix='aug_train', save_format='jpeg'):
        i += 1
        if i >= num_train_images:
            break 

    num_test_images = 5
    j = 0
    for batch in datagen.flow(image_array, batch_size=1, save_to_dir=test_save_dir, save_prefix='aug_test', save_format='jpeg'):
        j += 1
        if j >= num_test_images:
            break  
    print(f"Augmentation complete for image {num}")