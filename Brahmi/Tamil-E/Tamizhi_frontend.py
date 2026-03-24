import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.uint8)
    return image

def segment_characters_without_dots(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    character_images = []
    #bounding box
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 50 or w < 3 or h < 10:
            continue

        char_image = image[y:y+h, x:x+w]
        character_images.append((x, char_image))

    character_images = sorted(character_images, key=lambda elem: elem[0])
    character_images = [img[1] for img in character_images]
    return character_images

def plot_characters_dynamically(character_images):
    num_chars = len(character_images)
    if num_chars == 0:
        print("No characters detected.")
        return

    cols = 5  
    rows = (num_chars + cols - 1) // cols 

    fig, axs = plt.subplots(rows, cols, figsize=(12, 12), gridspec_kw={'wspace': 0.3, 'hspace': 0.3})
    axs = axs.flatten()

    for i, char_image in enumerate(character_images):
        axs[i].imshow(char_image, cmap='gray')
        axs[i].axis('off')  

    for j in range(i + 1, rows * cols):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def predict_character_sequence(image, model, class_labels):
    character_images = segment_characters_without_dots(image)
    predicted_sequence = []
    
    for char_img in character_images:
        char_img = cv2.resize(char_img, (64, 64))  
        char_img = char_img.astype('float32') / 255.0
        char_img = np.expand_dims(char_img, axis=-1)  
        char_img = np.expand_dims(char_img, axis=0)   

        predictions = model.predict(char_img)
        predicted_class = np.argmax(predictions)
        predicted_character = class_labels[predicted_class]
        predicted_sequence.append(predicted_character)

    sequence = ' '.join(predicted_sequence)
    print(f"Predicted Sequence: {sequence}")
    return sequence



file_path = r"E:\Capstone\Capstone_Tamizhi\Unicode_map - unicode_map (2).csv"
try:
    data = pd.read_csv(file_path, header=None)
except FileNotFoundError:
    print(f"File not found at {file_path}")
    raise
except pd.errors.EmptyDataError:
    print("The CSV file is empty.")
    raise

test_folder_path = r"E:\Capstone\Capstone_Tamizhi\archive\test"
class_labels = sorted(os.listdir(test_folder_path))

data.columns = ['Class Name', 'Unicode']
data['Unicode'] = data['Unicode'].astype(str).str.strip()
unicode_map = {}
for _, row in data.iterrows():

    class_name = str(row['Class Name'])
    unicode_value = row['Unicode']

    try:
        combined_character = "".join(
            chr(int(code_point[2:], 16)) for code_point in unicode_value.split() if code_point.startswith('U+')
        )
        unicode_map[class_name] = combined_character
    except ValueError:
        print(f"Invalid Unicode format for class {class_name}: '{unicode_value}'")
        
def get_unicode_sequence(class_names):
    class_name_list = class_names.split()
    characters = [unicode_map.get(name, "?") for name in class_name_list]
    return " ".join(characters)



model = tf.keras.models.load_model(r"E:\Capstone\Capstone_Tamizhi\tamizhi_ocr_model.h5")

#streamlit
st.set_page_config(page_title="TAMIZHI", layout="centered")
st.markdown(
    f"""
    <style>
        .title-style {{
            font-size: 100px;  /* Increased font size */
            font-family: 'Noto Serif Tamil', serif;  /* Set a font that supports Tamil */
            color: #6B4226;
            text-align: center;
            padding-top: 20px;
            padding-bottom: 20px;
            font-weight:bold;
        }}
        .output-style {{
            font-size: 24px;
            font-family: 'Georgia', serif;
            color: #4E342E;
            text-align: center;
            border: 2px solid #4E342E;
            padding: 20px;
            margin-top: 20px;
            border-radius: 15px;
            background-color: #FAF3E0;
        }}
        .upload-box {{
            border: 2px solid #8B5A2B;
            padding: 10px;
            border-radius: 10px;
        }}
    </style>
    <div class="background"></div>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="title-style">தமிழ்-E</div>', unsafe_allow_html=True)
st.write("Upload an image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="upload", help="Upload your image here", label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    def process_image(image):
     image = np.array(image)

     if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

     character_images = segment_characters_without_dots(image)
     plot_characters_dynamically(character_images)
    
     predicted_sequence = predict_character_sequence(image, model, class_labels)
     unicode_sequence = get_unicode_sequence(predicted_sequence)
    
     return unicode_sequence

    result_sentence = process_image(image)
    st.markdown(f'<div class="output-style">{result_sentence}</div>', unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #8B5A2B;'>", unsafe_allow_html=True)
