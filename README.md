# Introduction
### Naive Bayes
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.For some types of probability models, naive Bayes classifiers can be trained very efficiently in a supervised learning setting. In many practical applications, parameter estimation for naive Bayes models uses the method of maximum likelihood; in other words, one can work with the naive Bayes model without accepting Bayesian probability or using any Bayesian methods.

### Bag of Words
A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms.
The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents.
A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:<br>
1- A vocabulary of known words.<br>
2- A measure of the presence of known words.

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document.

Here we apply both algorythms in order to predict 3 types of news based on their description and headline.

## Dataset
The dataset for training and validation is available [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/data.csv)
The dataset for testing is available [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/test.csv)
The output of implemented model with the accuracy of 95% is provided [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/output.csv)

# Requirement
* Python > 3.7
* Jupyter Notebook
* PyTorch 1.6.0
* NLTK 3.5
* Numpy 1.19
* Pandas 1.1.2
* Matplotlib 3.3.2

# Implementation
The project has been splited to two different phase. In both phases, the formula of BAYESIAN-NETWORKS implemented for word counting and classifications.

![BAyesian nets Formula](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Bayesian%20Formula.png)
We should note that In the picture abow:
posterior probability is the probability that a news with the word xi is in the category of c. likelihood is the probability that the word xi will be in a news of category c, which will be computed as the total number of xi in category c news divided by the number of c category news. class probability is the number of news for each category divided by the total number of news. And at last, evidence is the probability that the word xi appears in any news regardless of its category. This probability will not be computed directly.
#### Note:
p(c|xi) or We need the probability of p(c|X) where X is a combination of xis which means the probability that a news that contains the words x1 to xn belongs to category c. This probability can be computed with the second formula in the picture below. The category with the highest p(c|X) for a given news is the category predicted by the model for that news.

## Preprocessing
The best approach for this project though is stemming approach. Because here we use bag of words, lemitization reduces accuracy when it doesn't know about the role of word in sentences and prematurely thinks that all words are nounes. of course before that, we tokenize our words by removing punctuations, stopwords, numbers, and make all letters in words lowercase. if we do not do that, it may reduce the accuracy since words won't be found in word probability dictionaty and also stopwords. Some treat these two as same. Actually, lemmatization is preferred over Stemming because lemmatization does morphological analysis of the words.

## Category Probabilities and Oversampeling
Here we first find the probabilities for each category, and then make copy of random rows in order to have rows in the level of 9000. This could prevent our model from differing accuracies along with three categories and reduces accuracy diffrence between 3 types.
### SMOT Oversampeling: 
I used SMOT Oversampling; wich is The most naive strategy and it is to generate new samples by randomly sampling with replacement the current available samples. And at last, I updated indexes to have ordinarily and unique index numbers.

# Results
During the phase 1, the model learns to differ the cattegory of "Business News" from "Travel News" by using preprocessed news' descriptions and titles. The algorythm implied was the Bag of words and TF-IDF. The result have shown bellow:

![Phase1](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase1%20Result.png)

In Phase2, the model tried to predict 3 classes of news all together. Consequently, the classification scores and the confusion matrix are illustrated in the following:

![Phase2](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase2%20Result.png)
![Phase2_CM](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase2%20CM.png)

Moreover, You can see all reports and code in jupyter notebook in eithor [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Bayesian%20Nets%20for%20TF-IDF.ipynb) as ipynb or [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Report.html) as HTML in your local computer.

# Questions & Discussion

1- tf–idf or TFIDF is the short for term frequency–inverse document frequency. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. tf–idf is one of the most popular term-weighting schemes today. As a matter of fact we used the described approach in our model. Previousley on the main preprocess code section, I tried to make the words for each dictionary unique and one in whole (by transform it to set and then bringing back to list again). using this Technique (adding repeated words) improved our model accuracy over 10 percent.

2- Precision is the result of deviding True Positive to the sum of True Positive + False Positive. This kind of evaluation does not consider negativeley predicted values. This could cause Serious problems when False Negatives are very Crucial for us; for example if we want to know that if a buiding is safe and strong or not; The precision could give us a good evaluation if we predict good conditioned buildings very well, but as for the bad ones, our model could brings us to Serious troubles.

3- If one word like "Tabriz" has repeated in just one datframes, since we find the probability of tabriz on that dataframe, it has the value of 1 devided to size of dataframe; which is most probably a low value. One the other hand, since there is no word "Tabriz" in other datasets, this value will be added in just one score of probability and lowers the probability of that currect category. So this is not the condition that we want to Experience! Thats why I Commented out the parts in phase 1 and phase 2 and Test Part in order to just compaire the words that are accuring in all kinds of categories. Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other. Here We Implemented these algorithms in order to prove that this is a very simple and precise way of solving problems with medium dificulty.

GOOD LUCK!
