# Using machine learning and handling imbalanced classes

After reading [http://fsecurify.com/using-machine-learning-detect-malicious-urls/](http://fsecurify.com/using-machine-learning-detect-malicious-urls/) and taking a look at the original dataset I noticed that the classes were imblanaced. Roughly 11% of the data was class 1 so a classifier could get an 88% accuracy score by just always predicting class 0. Class imbalance can be handled in sklearn via the `class_weight` parameter in the constructor of the classifiers. Setting it to `balanced` sklearn will set the weights automatically with the following formula:

```
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class 
frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
```

I was also interested in evaluating the performance of a few other classifiers. I also added more malicious URLs from [phishtank](https://www.phishtank.com/developer_info.php)

# Findings

Logisitic Regression does indeed have 98% accuracy. However, looking at the confusion matrix we can see that it also has a fairly high false negative. For this type of a classifier, it would arguable be best to minimize the false negative rate.

```
LogisticRegression
Accuracy score:  0.985791795152
ROC AUC:  0.984460239161
confusion matrix:  
 [[84098   977]
 [  843 42177]]
true negative:  84098
false positive:  977
false negative:  843
true positive:  42177
```

Random Forest gets a little bit lower accuracy score of 96.7%, but the false negatives are over 3 times lower than in the Logistic Regression.

```
RandomForest
Accuracy score:  0.967680237324
ROC AUC:  0.974134523603
confusion matrix:  [[81202  3873]
 [  267 42753]]
true negative:  81202
false positive:  3873
false negative:  267
true positive:  42753
```





