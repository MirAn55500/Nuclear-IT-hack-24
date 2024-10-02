import streamlit as st
import numpy as np
import json

from utils.preprocessing import dataset_to_text, to_lowercase, remove_stopwords
from utils.preprocessing import lemmatize, lemmatize_texts, lemma_replacement

from utils.embeddings import get_word_embedding, normalize_embeddings
from utils.embeddings import get_sentence_embedding, get_embeddings

from utils.clustering import use_agglomerative, use_dbSCAN, print_clusters
from utils.clustering import select_most_frequent_word, replace_keys_by_function, get_final_dict

from utils.visualizing import generate_word_cloud, plot_word_histogram 

def process_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    
    question, texts = dataset_to_text(data)
    texts = list(map(to_lowercase, texts))
    texts = list(map(remove_stopwords, texts))
    
    lemmatize_dict, lemmatized_texts = lemmatize_texts(texts)
    
    embeddings = get_embeddings(lemmatized_texts)

    word_clusters = use_agglomerative(embeddings, lemmatized_texts)
    final_word_dict = get_final_dict(word_clusters)
    final_word_dict = replace_keys_by_function(final_word_dict, lemma_replacement)
    
    return final_word_dict

st.title('Analyze Texts')

uploaded_file = st.file_uploader("Upload a JSON", type="json")

if uploaded_file:
    if st.button('Analyze'):
        with st.spinner('Processing...'):
            final_word_dict = process_data(uploaded_file)

        st.success('Done!')

        st.subheader('Visualization')
        word_cloud_image = generate_word_cloud(final_word_dict)
        st.pyplot(word_cloud_image)

        st.subheader('Histogram')
        word_histogram = plot_word_histogram(final_word_dict)
        st.pyplot(word_histogram)
