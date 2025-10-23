# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Load or train model
@st.cache_resource
def load_model():
    # Try to load existing model
    try:
        model = tf.keras.models.load_model('mnist_model.h5')
    except:
        # Train a new model if none exists
        st.info("Training new model... This may take a minute.")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)
        model.save('mnist_model.h5')
    
    return model

def preprocess_image(image):
    """Preprocess uploaded image to MNIST format"""
    # Convert to grayscale
    img = image.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert if necessary (MNIST has white digits on black background)
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Main app
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) or draw one!")

# Load model
model = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Upload Image", "Draw Digit", "Model Info"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Preprocess and predict
        processed_img = preprocess_image(image)
        
        with col2:
            st.image(
                processed_img[0], 
                caption='Preprocessed (28x28)', 
                use_container_width=True,
                clamp=True
            )
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Display results
        st.success(f"### Predicted Digit: **{predicted_class}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")
        
        # Show all probabilities
        st.write("#### All Class Probabilities:")
        prob_df = {
            'Digit': list(range(10)),
            'Probability': [f"{p*100:.2f}%" for p in prediction[0]]
        }
        st.bar_chart(prediction[0])

with tab2:
    st.write("### Draw a digit using your mouse/finger")
    st.info("This feature requires streamlit-drawable-canvas library")
    
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            if st.button("Predict Drawn Digit"):
                # Get canvas data
                img_data = canvas_result.image_data
                
                # Convert to grayscale
                img_gray = cv2.cvtColor(img_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
                
                # Resize to 28x28
                img_resized = cv2.resize(img_gray, (28, 28))
                
                # Normalize
                img_normalized = img_resized.astype('float32') / 255.0
                img_normalized = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                prediction = model.predict(img_normalized, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                st.success(f"### Predicted: **{predicted_class}** ({confidence*100:.2f}%)")
                st.bar_chart(prediction[0])
                
    except ImportError:
        st.warning("Install streamlit-drawable-canvas: `pip install streamlit-drawable-canvas`")

with tab3:
    st.write("### Model Architecture")
    st.code("""
    Sequential Model:
    - Input: 28x28 grayscale images
    - Flatten layer
    - Dense(128, activation='relu')
    - Dropout(0.2)
    - Dense(10, activation='softmax')
    
    Total params: ~101,770
    """)
    
    st.write("### Dataset Information")
    st.write("""
    - **Dataset**: MNIST Handwritten Digits
    - **Training samples**: 60,000
    - **Test samples**: 10,000
    - **Classes**: 10 (digits 0-9)
    - **Image size**: 28x28 grayscale
    """)
    
    st.write("### Tips for Best Results")
    st.write("""
    - Use clear, centered digits
    - White digit on dark background works best
    - Avoid overly stylized fonts
    - Ensure digit fills most of the image
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & TensorFlow | MNIST Classifier Demo")
