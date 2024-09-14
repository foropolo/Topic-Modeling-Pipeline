import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.util import ngrams
from collections import Counter
from gensim.models.phrases import Phrases, Phraser
import random



# Preprocess the text data
def preprocess(text):
    # Remove the part after #
    text = text.split('#')[0]
    # Tokenize the text
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (optional)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Zα-ωΑ-ΩάέήίΰαάέήίϊΰϊϋόύώΑΆΈΉΊΌΎΏ\s]', '', text)  # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    tokens = re.findall(r'\b\w+\b', text.lower())
    # Remove stop words
    tokens = [word for word in tokens if word not in greek_stopwords]
    return tokens


# Function to create dictionary and corpus
def create_dict_corpus(tokens):
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]
    return dictionary, corpus


# Function to perform LDA
def perform_lda(corpus, dictionary, num_topics, passes=10):
    lda_model = models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    return lda_model


# Function to evaluate the model
def evaluate_model(lda_model, tokens, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    return coherence_score

# Function to generate word clouds
def draw_wordcloud(lda_model, topic, num_words, title=''):
    words = lda_model.show_topic(topic, num_words)
    word_dict = {word: weight for word, weight in words}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_dict)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    #plt.show()
    plt.savefig(f'word_cloud_diagrams/word_cloud_{title}_{number_topics}.png',format= 'png')
    plt.close()

# Generate word clouds for each topic
def generate_wordclouds(lda_model, num_topics, title_prefix,num_words):
    for topic in range(num_topics):
        draw_wordcloud(lda_model, topic,num_words, title=f'{title_prefix} Topic {topic + 1}')


# Extract results to a text file
def extract_results(lda_model, num_topics, coherence_score, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(f'Coherence Score: {coherence_score}\n\n')
        for topic in range(num_topics):
            words = lda_model.show_topic(topic, 20)
            file.write(f'Topic {topic + 1}:\n')
            file.write(', '.join([word for word, weight in words]) + '\n\n')


if __name__ == '__main__':


    #Setters
    number_topics=5
    number_words = 10
    print("number of topics: ",number_topics)
    print("number of words: ",number_words)

    # Ensure you have the necessary NLTK data
    #nltk.download('stopwords')

    # Load the Excel file
    df = pd.read_excel('C:/Users/Foro/personal_projects/twitter-excel/Elliniki_lisi.xlsx')

    # Load Greek stop words from the text file
    with open('C:/Users/Foro/personal_projects/twitter-excel/greek_stopwords_with.txt', 'r', encoding='utf-8') as file:
        greek_stopwords = file.read().splitlines()

    # Strip any extra whitespace characters from the stop words
    greek_stopwords = [word.strip() for word in greek_stopwords]


    # Apply preprocessing
    df['tokens'] = df['text'].apply(preprocess)

    # Generate bigrams and trigrams
    bigram = Phrases(df['tokens'], min_count=5, threshold=100)
    trigram = Phrases(bigram[df['tokens']], threshold=100)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    df['bigrams'] = df['tokens'].apply(lambda tokens: bigram_mod[tokens])
    df['trigrams'] = df['bigrams'].apply(lambda tokens: trigram_mod[tokens])


    # Create dictionaries and corpora for unigrams, bigrams, and trigrams
    dictionary_unigram, corpus_unigram = create_dict_corpus(df['tokens'])
    dictionary_bigram, corpus_bigram = create_dict_corpus(df['bigrams'])
    dictionary_trigram, corpus_trigram = create_dict_corpus(df['trigrams'])


    # Perform LDA for unigrams, bigrams, and trigrams
    lda_model_unigram = perform_lda(corpus_unigram, dictionary_unigram,number_topics)
    lda_model_bigram = perform_lda(corpus_bigram, dictionary_bigram,number_topics)
    lda_model_trigram = perform_lda(corpus_trigram, dictionary_trigram,number_topics)


    # Evaluate models
    coherence_unigram = evaluate_model(lda_model_unigram, df['tokens'], dictionary_unigram)
    coherence_bigram = evaluate_model(lda_model_bigram, df['bigrams'], dictionary_bigram)
    coherence_trigram = evaluate_model(lda_model_trigram, df['trigrams'], dictionary_trigram)

    print(f'Unigram Coherence Score: {coherence_unigram}')
    print(f'Bigram Coherence Score: {coherence_bigram}')
    print(f'Trigram Coherence Score: {coherence_trigram}')



    generate_wordclouds(lda_model_unigram, lda_model_unigram.num_topics, 'Unigram', number_words)
    generate_wordclouds(lda_model_bigram, lda_model_bigram.num_topics, 'Bigram', number_words)
    generate_wordclouds(lda_model_trigram, lda_model_trigram.num_topics, 'Trigram', number_words)

    # Generate a word cloud for the whole dataset
    all_tokens = [token for tokens in df['trigrams'] for token in tokens]

    # Randomly select half of the tokens
    half_tokens = random.sample(all_tokens, len(all_tokens) // 2)

    wordcloud_all = WordCloud(width=1600, height=800, background_color='white').generate(' '.join(half_tokens))

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud_all, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for the Whole Dataset')
    plt.savefig(f'word_cloud_diagrams/word_cloud_ALL_{number_topics}.png',format= 'png', dpi=300)
    plt.close()

    # Visualize the topics with pyLDAvis
    #pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model_unigram, corpus_unigram, dictionary_unigram)
    pyLDAvis.save_html(vis, f'LDA_visualization/LDA_Visualization_unigram_{number_topics}.html')
    vis = pyLDAvis.gensim_models.prepare(lda_model_bigram, corpus_bigram, dictionary_bigram)
    pyLDAvis.save_html(vis, f'LDA_visualization/LDA_Visualization_bigram_{number_topics}.html')
    vis = pyLDAvis.gensim_models.prepare(lda_model_trigram, corpus_trigram, dictionary_trigram)
    pyLDAvis.save_html(vis, f'LDA_visualization/LDA_Visualization_trigram_{number_topics}.html')

    extract_results(lda_model_unigram, lda_model_unigram.num_topics, coherence_unigram, f'result_topic/result_topic_unigram_{number_topics}.txt')
    extract_results(lda_model_bigram, lda_model_bigram.num_topics, coherence_bigram, f'result_topic/result_topic_bigram_{number_topics}.txt')
    extract_results(lda_model_trigram, lda_model_trigram.num_topics, coherence_trigram, f'result_topic/result_topic_trigram_{number_topics}.txt')
