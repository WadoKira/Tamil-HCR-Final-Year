# Tamil-E
# OVERVIEW
Tamil-E is an **OCR-based** pipeline designed to convert images of Tamil-Brahmi script into modern Tamil text. The system uses **image preprocessing, character segmentation, a custom-trained CNN OCR model**, and a mapping module to translate the detected Brahmi characters into their contemporary Tamil equivalents.

This project serves as a digital preservation and interpretation tool for ancient Tamil inscriptions, aiding researchers, linguists, and enthusiasts in understanding historical texts.

## Tamil-Brahmi to Modern Tamil Converter
The pipeline follows these stages:

- Input Image: Ancient inscription in Tamil-Brahmi script.

- Preprocessing: Binarization, denoising, rotation correction.

- Segmentation: Line and character segmentation with logic to merge Brahmi diacritics (dots, strokes).

- OCR Model: A CNN-based classifier trained to detect individual Brahmi characters (325 classes).

- Postprocessing & Mapping: Maps Brahmi characters to their modern Tamil equivalents.

- Output: Cleaned, readable modern Tamil text.

---

# TECHNOLOGIES USED
## Image Processing
**OpenCV**: For grayscale conversion, contour detection, and segmentation.

**NumPy**: Pixel-based manipulations and preprocessing.

## OCR Model
**TensorFlow / Keras**: For training a CNN classifier on Brahmi character dataset.

**Conv2D + MaxPooling2D layers**: Used in a sequential CNN architecture.

## Mapping & Translation
**Custom mapping dictionary**: Brahmi-to-Tamil character mappings.

**Postprocessing logic**: Reconstructs Tamil words in reading order with dot and modifier handling.

## Interface 
**Streamlit**: To provide a user-friendly web UI for uploading images and seeing converted text.

---

# OUR PROJECT IN ACTION
Sample Input images (Tamil-Brahmi inscription):

![sirappupaayiram](https://github.com/user-attachments/assets/1d1f1c37-29da-46ce-88d7-5c1ce4d0a4eb)

![vijasarana](https://github.com/user-attachments/assets/6247b4a1-357f-4b30-8f96-beb1586f1d1e)

Here are some snapshots of our project outcomes:

![op1](https://github.com/user-attachments/assets/74cbfb04-a942-4bd9-9417-fdfafcab448a)

![op2](https://github.com/user-attachments/assets/8dc651cc-282e-4271-b595-ea03813679c1)
