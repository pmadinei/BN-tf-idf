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

# Implementation and Results
The project has been splited to two different phase. In both phases, the formula of BAYESIAN-NETWORKS implemented for word counting and classifications.

![BAyesian nets Formula](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Bayesian%20Formula.png)

During the phase 1, the model learns to differ the cattegory of "Business News" from "Travel News" by using preprocessed news' descriptions and titles. The algorythm implied was the Bag of words and TF-IDF. The result have shown bellow:

![Phase1](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase1%20Result.png)

In Phase2, the model tried to predict 3 classes of news all together. Consequently, the classification scores and the confusion matrix are illustrated in the following:

![Phase2](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase2%20Result.png)
![Phase2_CM](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Phase2%20CM.png)

Moreover, You can see all reports and code in jupyter notebook in eithor [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Bayesian%20Nets%20for%20TF-IDF.ipynb) as ipynb or [HERE](https://github.com/pmadinei/BN-tf-idf/blob/master/Docs/Report.html) as HTML in your local computer.
