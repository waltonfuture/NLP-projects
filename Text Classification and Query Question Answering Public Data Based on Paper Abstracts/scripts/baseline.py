import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('../data/train.csv', sep=',')

test_df = pd.read_csv('../data/test.csv', sep=',')

train_df.head()

test_df.head()

train_df['Topic(Label)'], lbl = pd.factorize(train_df['Topic(Label)'])

train_df['Title'] = train_df['Title'].apply(lambda x: x.strip())
train_df['Abstract'] = train_df['Abstract'].fillna('').apply(lambda x: x.strip())
train_df['text'] = train_df['Title'] + ' ' + train_df['Abstract']
train_df['text'] = train_df['text'].str.lower()

test_df['Title'] = test_df['Title'].apply(lambda x: x.strip())
test_df['Abstract'] = test_df['Abstract'].fillna('').apply(lambda x: x.strip())
test_df['text'] = test_df['Title'] + ' ' + test_df['Abstract']
test_df['text'] = test_df['text'].str.lower()

tfidf = TfidfVectorizer(max_features=2500)

#----------------train---------------

train_tfidf = tfidf.fit_transform(train_df['text'])
clf = SGDClassifier()
cross_val_score(clf, train_tfidf, train_df['Topic(Label)'], cv=5)

test_tfidf = tfidf.transform(test_df['text'])
clf = SGDClassifier()
clf.fit(train_tfidf, train_df['Topic(Label)'])
test_df['Topic(Label)'] = clf.predict(test_tfidf)


#----------------output----------------
test_df['Topic(Label)'] = test_df['Topic(Label)'].apply(lambda x: lbl[x])
test_df[['Topic(Label)']].to_csv('submit.csv', index=None)