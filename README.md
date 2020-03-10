# K-NN Algorithm and Linear Regression Model


## Getting Started

### Installing Python
To use the script, you will need to install Python 3.6.x and add to path:
- [Python 3.6.x](https://www.python.org/downloads/release/python-360/)


### Installing Dependencies
After cloning the project, go to the root directory:

Install the dependent libraries by typing the following in cmd/terminal:
```
$ pip install -r requirements.txt
```

### Starting the Script
To run the script, go to the root directory and run `python` in cmd/terminal and type the following in the python console:
```
>>> from knn_main import *
```
*Note: ensure that python refers to Python 3.6.x*

### Running K-NN on Regression Dataset (Q1)
To run the k-nn algorithm on regression dataset [mauna_loa,rosenbrock,pumadyn32nm], type the following in the python console:
```
>>> run_Q1(k_range=[1,31])
```
*Note: k_range takes a list containing lower bound and upper bound of k values*


### Running K-NN on Classification Dataset (Q2)
To run the k-nn algorithm on classification dataset [iris,mnist_small], type the following in the python console:
```
>>> run_Q2(k_range=[1,31])
```
*Note: k_range takes a list containing lower bound and upper bound of k values*

### Running K-NN with KD Tree and Compare Performance (Q3)
Type the following in the python console:
```
>>> run_Q3(d=list(range(2,10)))
```
*Note: d takes a list containing values of dimension numbers*

### Running Linear Regression with SVD on All Dataset (Q4)
Type the following in the python console:
```
>>> run_Q4()
```




## Built With
* [numpy](https://numpy.org/) - all variables are numpy arrays

* [sklearn](https://scikit-learn.org/stable/) - kd tree data structure
