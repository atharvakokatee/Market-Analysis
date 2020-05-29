import json
import time
import random

tic = time.time()

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_destribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

file_name = './data/sentiment/Books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))

from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_destribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_destribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

# Liner SVM
from sklearn.svm import SVC
clf_svm = SVC(kernel='rbf')
clf_svm.fit(train_x_vectors,train_y)

y_pred = clf_svm.predict(test_x_vectors)
print(y_pred)

print("SVC: " + str(clf_svm.score(test_x_vectors, test_y)))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

y_pred = clf_dec.predict(test_x_vectors)
print(y_pred)

print("DTC: " + str(clf_dec.score(test_x_vectors, test_y)))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)

y_pred = clf_gnb.predict(test_x_vectors)
print(y_pred)

print("NB: " + str(clf_gnb.score(test_x_vectors, test_y)))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

y_pred = clf_log.predict(test_x_vectors)
print(y_pred)

print("LGR: " + str(clf_log.score(test_x_vectors, test_y)))

# Mean Accuracy
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))

# F1 Scores
from sklearn.metrics import f1_score
f1score = f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print(f1score)

test_set = ["very fun", "bad book do not buy", "horrible waste of time"]
new_test = vectorizer.transform(test_set)
y_pred = clf_svm.predict(new_test)
print(y_pred)

# Hyperarameter Tuning
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}
svc = SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
print(clf)
print(clf.score(test_x_vectors, test_y))

# Saving Model
import pickle
with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Load Model
with open('./models/sentiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)
print(test_x[0])
print(loaded_clf.predict(test_x_vectors[0]))

toc = time.time()

print("Total time: " + str(1000*(toc-tic)) + "ms")
