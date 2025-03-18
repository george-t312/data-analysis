# Import Dependencies
import pandas as pd
import numpy as np
import re
import spacy
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim import corpora, models
import gensim

# Download necessary NLTK resources 
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("wordnet")

# Load dataset
file_path = "./models/complaints_processed_short.csv"
df = pd.read_csv(file_path)

# Load spacy model for NER
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing step
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize
    return " ".join(words)

df["clean_text"] = df["narrative"].astype(str).apply(clean_text)

# TF-IDF vectorization
vectorizer_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer_tfidf.fit_transform(df["clean_text"])
tfidf_features = vectorizer_tfidf.get_feature_names_out()

# Word2Vec training
sentences = [text.split() for text in df["clean_text"]]
word2vec_model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)

# Topic extraction with LDA
texts = [text.split() for text in df["clean_text"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Evaluate LDA model evaluation
coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model_lda.get_coherence()
print(f"LDA Model Coherence Score: {coherence_score}\n")

# Extract and save LDA topics
def get_lda_topics(model, num_words=5):
    topics = {}
    topic_names = []

    for i, topic in model.show_topics(formatted=False, num_words=num_words):
        top_words = [word for word, _ in topic]
        topic_label = "_".join(top_words[:3])  # Auto-label topics
        topic_names.append(topic_label)
        topics[topic_label] = top_words

    topics_df = pd.DataFrame(topics)
    print("LDA Topics:\n")
    print(topics_df)
    
    topics_df.to_csv("./results/lda_topics.csv", index=False)
    return topic_names

lda_topic_labels = get_lda_topics(lda_model)

# LSA
lsa_model = TruncatedSVD(n_components=5, random_state=42)
lsa_topics = lsa_model.fit_transform(X_tfidf)

# Extract and save LSA topics
def get_lsa_topics(model, feature_names, num_words=5):
    topics = {}
    topic_names = [] 

    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topic_label = "_".join(top_words[:3])  # Auto-label topics
        topic_names.append(topic_label)
        topics[topic_label] = top_words

    topics_df = pd.DataFrame(topics)
    print("LSA Topics:\n")
    print(topics_df)

    topics_df.to_csv("./results/lsa_topics.csv", index=False)
    return topic_names

lsa_topic_labels = get_lsa_topics(lsa_model, vectorizer_tfidf.get_feature_names_out())

# NER
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df["entities"] = df["clean_text"].apply(extract_entities)

# Visualize LDA topics (wordcloud)
def plot_wordcloud(topic_words, filename="./results/visualized_lda_topics.png"):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(topic_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("LDA Topics")
    plt.savefig(filename)
    plt.close()

# Generate word cloud for LDA topics
top_lda_words = [word for sublist in lda_topic_labels for word in sublist.split("_")]
plot_wordcloud(top_lda_words, "./results/visualized_lda_topics.png")

# Visualize LSA topics (heatmap)
def plot_lsa_heatmap(lsa_topics, filename="./results/visualized_lsa_topics.png"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(lsa_topics, cmap="coolwarm", annot=False)
    plt.title("LSA Topics")
    plt.xlabel("Topics")
    plt.ylabel("Documents")
    plt.savefig(filename)
    plt.close()

plot_lsa_heatmap(lsa_topics, "./results/visualized_lsa_topics.png")

# Store processed data
df.to_csv("./results/processed_data.csv", index=False)
print("[+] Data analysis completed sucessfully, results can be found in the results/ folder.")
