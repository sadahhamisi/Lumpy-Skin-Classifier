import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('CNN.h5')

# Define the class names for predictions
class_names = ["Healthy Cow", "Lumpy Cow"]

    # Set Streamlit app configuration
st.set_page_config(page_title='Disease Classification', layout='wide')

# Rest of the code remains the same
def Predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class,


# Define the main content of the app
def main():
    # Set app title and description
    st.title('Lumpy Skin Disease Classification')
    st.write('Upload an image to classify.')

    # Create a file uploader
    uploaded_file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

    # Perform prediction when file is uploaded
    if uploaded_file is not None:
        # Read the image and preprocess it
        img = tf.keras.preprocessing.image.load_img(uploaded_file)
        img = tf.image.resize(img, (256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Make the prediction using the loaded model
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Display the predicted class and confidence
        st.subheader('Prediction Result')
        st.write('Class:', predicted_class)



        st.subheader('click the link to know more')
        st.write('https://cow-disease-diagnosis.vercel.app/')

# Run the app
if __name__ == '__main__':
    main()
