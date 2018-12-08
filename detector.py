import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV,cross_val_score
import seaborn as sns

import nltk
nltk.download('stopwords')

data_frame = pd.read_table('SMSSpamCollection',header=None)

#print(data_frame.head())
#print(data_frame.info())

y = data_frame[0]
#print(y.value_counts())

label_encoder = LabelEncoder()
y_encoder = label_encoder.fit_transform(y)
#print(y_encoder)
#print(data_frame)

message = data_frame[1]
#print(len(message))

processed = message.str.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b','emailaddr')
processed = processed.str.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)','httpaddr')
processed = processed.str.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()

stop_words = nltk.corpus.stopwords.words('english')

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in set(stop_words)))

porter = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(porter.stem(term) for term in x.split()))

def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
    cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
                     cleaned)
    cleaned = re.sub(
        r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        'phonenumbr', cleaned)
    cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
    cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
    return ' '.join(
        porter.stem(term)
        for term in cleaned.split()
        if term not in set(stop_words)
    )

#print(processed == message.apply(preprocess_text)).all()

#example = "congratl numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send messag emailaddr get ticket"
#preprocess_text(example)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(processed)

#print(X_ngrams.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_ngrams,
    y_encoder,
    test_size=0.2,
    random_state=42,
    stratify=y_encoder
)

clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#print(metrics.f1_score(y_test, y_pred))

#print(pd.DataFrame(
#    metrics.confusion_matrix(y_test, y_pred),
#    index=[['actual', 'actual'], ['spam', 'ham']],
#    columns=[['predicted', 'predicted'], ['spam', 'ham']]
#))

sample_space = np.linspace(500, len(message) * 0.8, 10, dtype='int')

train_sizes, train_scores, valid_scores = learning_curve(
    estimator=svm.LinearSVC(loss='hinge', C=1e10),
    X=X_ngrams,
    y=y_encoder,
    train_sizes=sample_space,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=40),
    scoring='f1',
    n_jobs=-1
)

def make_tidy(sample_space, train_scores, valid_scores):
    messy_format = pd.DataFrame(
        np.stack((sample_space, train_scores.mean(axis=1),
                  valid_scores.mean(axis=1)), axis=1),
        columns=['# of training examples', 'Training set', 'Validation set']
    )

    return pd.melt(
        messy_format,
        id_vars='# of training examples',
        value_vars=['Training set', 'Validation set'],
        var_name='Scores',
        value_name='F1 score'
    )

#g = sns.FacetGrid(
#    make_tidy(sample_space, train_scores, valid_scores), hue='Scores', size=5
#)

#g.map(plt.scatter, '# of training examples', 'F1 score')
#g.map(plt.plot, '# of training examples', 'F1 score').add_legend();

param_grid = [{'C': np.logspace(-4, 4, 20)}]

grid_search = GridSearchCV(
    estimator=svm.LinearSVC(loss='hinge'),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores = cross_val_score(
    estimator=grid_search,
    X=X_ngrams,
    y=y_encoder,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
    scoring='f1',
    n_jobs=-1
)

#print(scores.mean())

grid_search.fit(X_ngrams, y_encoder)
final_clf = svm.LinearSVC(loss='hinge', C=grid_search.best_params_['C'])
final_clf.fit(X_ngrams, y_encoder);

#print(pd.Series(final_clf.coef_.T.ravel(),index=vectorizer.get_feature_names()).sort_values(ascending=False)[:20])

def spam_filter(message):
    if final_clf.predict(vectorizer.transform([preprocess_text(message)])):
        return 'spam'
    else:
        return 'not spam'

#print(spam_filter("congratl numbr ticket hamilton nyc httpaddr worth moneysymbnumbr call phonenumbr send messag emailaddr get ticket"))

os.system("clear")

while True:
    msg = raw_input("Enter your Message : ")
    print(spam_filter(msg))
