import streamlit as st
import os
import json

from utils.analyze_gigachat import analyze_word_frequency_gigachat
from utils.preprocessing import data_to_text, to_lowercase, remove_stopwords
from utils.preprocessing import lemmatize, lemmatize_texts, lemma_replacement

from utils.embeddings import get_word_embedding, normalize_embeddings
from utils.embeddings import get_sentence_embedding, get_embeddings

from utils.clustering import use_agglomerative, use_dbSCAN, print_clusters
from utils.clustering import select_most_frequent_word, replace_keys_by_function, get_final_dict

from utils.visualizing import generate_word_cloud, plot_word_histogram

def process_data(data: dict):
    question, texts = data_to_text(data)
    texts = list(map(to_lowercase, texts))
    
    lemmatize_dict, lemmatized_texts = lemmatize_texts(texts)
    
    embeddings = get_embeddings(lemmatized_texts)

    word_clusters = use_dbSCAN(embeddings, lemmatized_texts)
    final_word_dict = get_final_dict(word_clusters, lemmatize_dict)
    #final_word_dict = replace_keys_by_function(final_word_dict, lemma_replacement, lemmatize_dict)
    
    return final_word_dict

st.title('Анализатор опросов')

uploaded_file = st.file_uploader("Upload a JSON", type="json")

if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    content = json.loads(file_content)
    if st.button('Проанализировать'):
        with st.spinner('Обработка...'):
            final_word_dict = process_data(content)

        st.success('Сделано!')

        st.subheader('Облако слов:')
        word_cloud_image = generate_word_cloud(final_word_dict)
        st.pyplot(word_cloud_image)

        st.subheader('Распределение:')
        word_histogram = plot_word_histogram(final_word_dict)
        st.pyplot(word_histogram)

        if os.environ.get('GIGACHAT_ID'):
            text_answer = analyze_word_frequency_gigachat(final_word_dict, content['question'])
            st.write(text_answer)
