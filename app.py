import streamlit as st
from predict import load_model_and_tokenizer, predict_next_word

st.set_page_config(page_title="Next Word Prediction - PyTorch", page_icon="📚")
st.title("Next Word Prediction Using LSTM (PyTorch)")
st.caption("Trained on Shakespeare's Hamlet")

@st.cache_resource
def load_assets():
    return load_model_and_tokenizer()

model, tokenizer, max_sequence_len = load_assets()
text = st.text_input("Enter a sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    if not text.strip():
        st.warning("Please enter some words.")
    else:
        result = predict_next_word(text, model, tokenizer, max_sequence_len)
        st.success(f"Next word: {result}")
