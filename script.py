import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
import random
from urllib.parse import urlparse, parse_qs
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

np.random.seed(92016)
random.seed(92016)


def get_tokens(url):
    parts = urlparse(url)
    qs = parse_qs(parts.query)
    path = parts.path
    tokens = path.split('/')
    for k, v in qs.items():
        tokens.append(k)
        for item in v:
            tokens.append(item)
    tokens = [item.split('.') for item in tokens]
    return itertools.chain.from_iterable(tokens)


def print_stats(y, yhat, header):
    print(header)
    print('Accuracy score: ', accuracy_score(y, yhat))
    print('ROC AUC: ', roc_auc_score(y, yhat))
    print('confusion matrix: ', confusion_matrix(y, yhat))
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    print('true negative: ', tn)
    print('false positive: ', fp)
    print('false negative: ', fn)
    print('true positive: ', tp)
    print()


def main():
    allurlsdata = pd.read_csv('data/data_phish.csv')  # reading file

    allurlsdata = np.array(allurlsdata)
    random.shuffle(allurlsdata)

    y = allurlsdata[:, 1].astype(np.int8)
    corpus = allurlsdata[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(corpus, y, stratify=y, test_size=0.3, random_state=92016)

    lgs = make_pipeline(TfidfVectorizer(tokenizer=get_tokens),
                        LogisticRegression(penalty='l1', C=.99, random_state=42, n_jobs=-1, class_weight='balanced'))
    lgs.fit(X_train, y_train)

    pred = lgs.predict(X_test)

    print_stats(y_test, pred, 'LogisticRegressions')

    rf = make_pipeline(TfidfVectorizer(tokenizer=get_tokens),
                       RandomForestClassifier(n_estimators=100, min_samples_split=100, n_jobs=-1, random_state=92016,
                                              class_weight='balanced'))

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    print_stats(y_test, pred, 'RandomForest')

    # xgb_clf = make_pipeline(TfidfVectorizer(tokenizer=get_tokens),
    #                         xgb.XGBClassifier(learning_rate=.1, n_estimators=2000,
    #                                           max_depth=11, scale_pos_weight=8,
    #                                           seed=92016, min_child_weight=7,
    #                                           subsample=.8, colsample_bytree=.8))
    #
    # xgb_clf.fit(X_train, y_train)

    # pred = xgb_clf.predict(X_test)

    # print_stats(y_test, pred, 'XGBoost')

    # checking some random URLs. The results come out to be expected. The first two are okay and the last four are malicious/phishing/bad
    # X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
    # X_predict = vectorizer.transform(X_predict)
    # y_Predict = lgs.predict(X_predict)
    # print(y_Predict)	#printing predicted values


if __name__ == '__main__':
    main()
