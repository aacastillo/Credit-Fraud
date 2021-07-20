# Credit Fraud Detection with Machine Learning Anomaly Detection
### Project Description
I am using a credit card dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) with over 284,807 credit card transaction and too large to upload to GitHub with over 147MB. We will clean the data and do a data exploration to compare fraudulent and non-fraudulent transactions. With the clean data set we will run two anomaly detection machine learning algorithms: [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) and [Local Outlier Factor Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html). We will then test how these machine learning algorithms will perform with and without knowing the proportion of outliers in the dataset, called the contamination proportion.

### Machine Learning Algorithms
#### _Isolation Forest_
The Isolation Forest algorithm is a tree-based unsupervised machine learning algorithm that can efficiently isolate outliers. The algorithm was first published in 2012 and then incorporated into the Python Scikit-Learn library. It isolates the outliers by randomly selecting a feature from the given set of features and then randomly selecting a split value between the max and min values of that feature. This random partitioning of features will produce shorter paths in trees for the anomalous data points, thus distinguishing them from the rest of the data.

#### _Local Outlier Factor (LOF)_
LOF is an unsupervised machine learning algorithm, proposed in 2000, which produces an anomaly score that represents data points which are outliers in the data set. It does this by measuring the local density deviation of a given data point with respect to the data points near it. Think of it as a scatter plot with each transaction mapped onto the graph and using something similar to the K-Nearest Neighbors algorithm.

### Database Description
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds occuring in a total of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

The data contains only numerical input variables which are the result of a Principal Component Analysis (PCA) transformation. PCA is useful since the dimensions of the features we are dealing with are high. We are basically compressing the data and removing noise. However, there are two drawbacks from the dataset we are given. One, we dont know how much of the data was compressed. PCA works by taking the two dimensions that explains the highest amount of variability in the data. We are not told whether the PCA explains 90% of the variability in the data or 10%. Two, Unfortunately, due to confidentiality issues, the dataset does not provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.

Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

### Prerequsites
1. To run the code you must have Jupyter Notebooks, you can do a test run and upload the project on the browser for free [here](https://jupyter.org/).

### Dependancies
1. numpy library prior vs. of 
2. pandas library
3. scikit learn library, specifically the Isolation Forest algorithm and Local Outlier Factor
4. matplotlib
5. Kaggle credit card dataset

### To Do
1. The project uses a deprecated version of numpy which had several updates after v1.20
