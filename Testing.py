from keras.preprocessing import image
from keras.models import load_model
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

def classify_image(img_path, model):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    result = "Cat" if preds[0][0] > preds[0][1] else "Dog"
    confidence = max(preds[0])
    return result, confidence

def main():
    model = load_model('dog_cat_classifier.h5')
    img_path = input("Enter the path to the image: ")
    result, confidence = classify_image(img_path, model)
    print(f"The uploaded image is a {result} with confidence {confidence}")

if __name__ == '__main__':
    main()
