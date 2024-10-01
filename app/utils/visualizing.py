import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_word_cloud(word_dict: dict):
    """
    Create word cloud from word dictionary and return matplotlib figure.
    """
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          collocations=False).generate_from_frequencies(word_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_word_histogram(word_dict: dict):
    """
    Create word histogram and return matplotlib figure.
    """
    sorted_word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    
    words = list(sorted_word_dict.keys())
    frequencies = list(sorted_word_dict.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, frequencies, color='skyblue')
    ax.set_ylabel('Frequencies')
    ax.set_xlabel('Words and phrases')
    ax.set_title('Word frequencies')
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    return fig