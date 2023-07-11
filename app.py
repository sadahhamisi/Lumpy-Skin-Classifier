import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('CNN.h5')

# Define the class names for predictions
class_names = ["Healthy Cow", "Lumpy Cow"]

# Set Streamlit app configuration
st.set_page_config(page_title='Disease Classification', layout='wide')
# Define the login page
def login_page():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login", key="login_button")

    if login_button:
        # Validate the username and password
        if username == "your_username" and password == "your_password":
            st.sidebar.success("Logged in successfully!")
             # Addind a logout button
            st.sidebar.button("Logout", on_click=logout, key="logout_button") 
            st.sidebar.text("You are logged in as: " + username)
            st.experimental_set_query_params(logged_in="true")  # Set query parameter
            show_main_app()
        else:
            st.sidebar.error("Invalid username or password")

# Define the logout function
def logout():
    st.experimental_set_query_params(logged_in="false")
    st.sidebar.text("You have been logged out.")
    st.sidebar.empty()  # Clear the sidebar content
    st.sidebar.button("Login", on_click=login_page, key="login_button")  # Show login button

# Define the main app content
def show_main_app():
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

        if predicted_class == "Lumpy Cow":
            st.subheader('Click the link to know more')
            st.write('https://cow-disease-diagnosis.vercel.app/')

# Run the app
def main():
    # Check if the user is authenticated
    if st.experimental_get_query_params().get("logged_in", ["false"])[0] == "true":
        show_main_app()
    else:
        login_page()

if __name__ == '__main__':
    main()
