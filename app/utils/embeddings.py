import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

import numpy as np
from tqdm import tqdm

TYPE_EMBEDDING = 'ru-en-RoSBERTa'

if TYPE_EMBEDDING == 'word2ec':
    # Word2vec embeddings
    word2vec_model = KeyedVectors.load_word2vec_format("ruwiki_20180420_300d.txt")
else:
    # ru-en-RoSBERTa embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")
    model = AutoModel.from_pretrained("ai-forever/ru-en-RoSBERTa").to(device)


def get_word2vec_embedding(word: str) -> np.ndarray:
    """
    Get word2vec embedding for given word
    """
    words = word.split()

    return word2vec_model.get_vector(words[-1], norm=True).reshape(-1)



def get_word_embedding(word: str) -> torch.Tensor:
    """
    Get word embedding for given word
    """
    

    inputs = tokenizer("clustering: " + word, return_tensors="pt", max_length=512, padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]

def normalize_embeddings(embedding: torch.Tensor) -> torch.Tensor:
    """
    Normalize embeddings
    """
    return F.normalize(embedding, p=2, dim=1)

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """
    Get sentence embedding for given sentence
    """
    sentence_vector = normalize_embeddings(get_word_embedding(sentence)).cpu().numpy().reshape(-1)
    return sentence_vector

def get_embeddings(lemmatized_texts: list) -> np.ndarray:
    """
    Get embeddings for given texts
    """
    embeddings = []
    for sentence in tqdm(set(lemmatized_texts)):
        embeddings.append(get_sentence_embedding(sentence))

    embeddings = np.array(embeddings)
    return embeddings