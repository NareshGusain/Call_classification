import streamlit as st
import joblib


label = {0: 'Billing Inquiries', 1: 'Claims', 2: 'Complaints', 3: 'Feedback', 4: 'Product Inquiry', 5: 'Sales', 6: 'Technical Support'}
# Load pre-trained Passive-Aggressive Classifier and TF-IDF vectorizer
model = joblib.load('passive_aggressive_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

st.title("Text Classification App with Passive-Aggressive Classifier")


input_text = st.text_area("Enter text here:", "")

if st.button("Predict"):
    if input_text:
        # Vectorize the input text
        processed_text = vectorizer.transform([input_text])
        
        # Make prediction
        prediction = model.predict(processed_text)
        predicted_class = label[prediction[0]]  # Get the class label
        
        # Display the prediction
        st.write("Predicted Category:", predicted_class)
    else:
        st.write("Please enter some text for classification.")
