# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

# Cleaning the text sentences so that punctuation marks, stop words & digits are removed


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


print("There are 100 mails of following five classes on which K-NN classification and K-means clustering"
      " is performed : \n1. naver \n2. daum \n3. gmail \n4. tmax \n5. hotmail")
path = "Sentences.txt"

train_clean_sentences = []
fp = open(path, 'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)

# Creating true labels for 500 training sentences
y_train = np.zeros(500)
y_train[0:100] = 0
y_train[100:200] = 1
y_train[200:300] = 2
y_train[300:400] = 3
y_train[400:500] = 4

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X, y_train)

# Clustering the training 30 sentences with K-means technique
modelkmeans = KMeans(n_clusters=5, init='k-means++', max_iter=500, n_init=100)
modelkmeans.fit(X)


test_sentences = ["testtest@naver.com",
                  "leejingu@daum.net",
                  "backgu2002@gmail.com",
                  "lbe0522@tmax.co.kr",
                  "awjeifojawofji@hotmail.com"]

test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+", "", cleaned)
    test_clean_sentence.append(cleaned)

Test = vectorizer.transform(test_clean_sentence)

true_test_labels = ['naver', 'daum', 'gmail', "tmax", "hotmail"]
predicted_labels_knn = modelknn.predict(Test)
predicted_labels_kmeans = modelkmeans.predict(Test)

print("\nBelow 5 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ",
      test_sentences[0], "\n2. ", test_sentences[1], "\n3. ", test_sentences[2], "\n4. ", test_sentences[3], "\n5. ", test_sentences[4])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n", test_sentences[0], ":", true_test_labels[np.int64(
    predicted_labels_knn[0])],
    "\n", test_sentences[1], ":", true_test_labels[np.int64(
        predicted_labels_knn[1])],
    "\n", test_sentences[2], ":", true_test_labels[np.int64(
        predicted_labels_knn[2])],
    "\n", test_sentences[3], ":", true_test_labels[np.int64(
        predicted_labels_knn[3])],
    "\n", test_sentences[4], ":", true_test_labels[np.int64(
        predicted_labels_knn[4])])

print("\n-------------------------------PREDICTIONS BY K-Means--------------------------------------")
print("\nIndex of naver cluster : ", Counter(
    modelkmeans.labels_[0:100]).most_common(1)[0][0])
print("Index of daum cluster : ", Counter(
    modelkmeans.labels_[100:200]).most_common(1)[0][0])
print("Index of gmail cluster : ", Counter(
    modelkmeans.labels_[200:300]).most_common(1)[0][0])
print("Index of tmax cluster : ", Counter(
    modelkmeans.labels_[300:400]).most_common(1)[0][0])
print("Index of hotmail cluster : ", Counter(
    modelkmeans.labels_[400:500]).most_common(1)[0][0])

print("\n", test_sentences[0], ":", predicted_labels_kmeans[0],
      "\n", test_sentences[1], ":", predicted_labels_kmeans[1],
      "\n", test_sentences[2], ":", predicted_labels_kmeans[2],
      "\n", test_sentences[3], ":", predicted_labels_kmeans[3],
      "\n", test_sentences[4], ":", predicted_labels_kmeans[4],)
