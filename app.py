import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np 
import streamlit as st

# load model
model = load_model('next_word_lstm.h5')

# load tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[ -(max_sequence_len - 1): ] ## truncate token_list to fit input size
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word prediction with LSTM and Early Stopping")

input_text = st.text_input("Enter the sequence of words", "To be or not to be")

if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"<html><body><h2>{next_word}</h2></body></html>", unsafe_allow_html=True)
