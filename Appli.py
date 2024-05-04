import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Set the dataset directory
dataset_dir = 'E:/image_classification/dataset'

# Create the data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

val_generator = val_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'validation'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(dataset_dir, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_acc:.2f}')

# Save the trained model
model.save('dog_classifier_model.h5')

# Streamlit app
def home_page():
    st.title("Welcome to the Dog Image Classifier")
    st.write("This app allows you to classify images of dogs and learn about convolutional neural networks.")
    st.write("To get started, please select an option from the sidebar.")

def image_prediction_page():
    st.title("Image Classifier")

    dog_name = st.text_input("Enter the name of the dog:")

    if st.button("Classify"):
        # Load the image based on the dog's name
        image_path = os.path.join(dataset_dir, 'test', 'dogs', f"{dog_name}.jpg")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"Image of {dog_name}", use_column_width=True)

            # Preprocess the image and make a prediction
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            prediction = model.predict(image)[0][0]
            if prediction > 0.5:
                st.write(f"The image is of a dog with {(prediction * 100):.2f}% confidence.")
            else:
                st.write(f"The image is not of a dog with {(1 - prediction) * 100:.2f}% confidence.")
        else:
            st.write(f"Sorry, we don't have an image for the {dog_name} breed in our dataset.")

def cnn_explanation_page():
    st.title("Convolutional Neural Networks (CNNs) Explained")
    st.write("Convolutional Neural Networks (CNNs) are a type of deep learning algorithm that are particularly well-suited for image classification tasks.")
    st.write("CNNs work by extracting features from the input image through a series of convolutional and pooling layers. The convolutional layers apply filters to the image, which detect specific patterns or features, while the pooling layers reduce the spatial size of the feature maps, making the model more robust to small variations in the input.")
    st.write("The final layers of a CNN are typically fully connected layers, which take the extracted features and use them to classify the input image into one or more categories.")
    st.write("Some key advantages of CNNs for image classification include their ability to automatically learn relevant features from the data, their translation invariance (the ability to detect features regardless of their position in the image), and their scalability to large-scale datasets.")
    st.write("If you'd like to learn more about the technical details of CNNs, I recommend checking out the resources on the [Keras website](https://keras.io/guides/convolutional_layers/) or the [TensorFlow documentation](https://www.tensorflow.org/tutorials/images/cnn).")

# Set up the Streamlit app
PAGES = {
    "Home": home_page,
    "Image Prediction": image_prediction_page,
    "CNN Explanation": cnn_explanation_page,
}

def app():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[selection]()

if __name__ == "__main__":
    # Load the trained model
    model = load_model('dog_classifier_model.h5')
    app()


