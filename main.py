import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

# Download the necessary resources (you might have already done this)
nltk.download('punkt')
nltk.download('movie_reviews')

# Load the movie review dataset
movie_reviews.ensure_loaded()
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Define a feature extractor
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words.keys())[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

# Extract features and split the dataset
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[:1500], featuresets[1500:]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the model
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Classifier Accuracy:", accuracy)

# User input and sentiment analysis
user_input = input("Enter a movie review: ")
words = word_tokenize(user_input)
features = document_features(words)
sentiment = classifier.classify(features)
print("Sentiment:", sentiment)
