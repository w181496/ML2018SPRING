# Homework 2 -- Income Prediction

- Homework2 web page
    - https://ntumlta2018.github.io/ml-web-hw2/

- Logistic_regression.py
    - Logistic Regression Model
    - using all feature
    - Adagrad
    - Normalization
    - Regularization
    
- Generative.py
    - Probabilistic Generative Model
    - using all feature
    - Normalization

- model.npy
    - saving the result parameters of `Logistic_regression.py`

- g_model.npy
    - saving the result parameters of `Generative.py`

- hw2_logistic.py
    - using `Logistic_regression.py` result to estimate testing data

- hw2_logistic.sh
    - usage: `bash hw2_logistic.sh a b train_X train_Y test_X predict.csv`

- hw2_generative.py
    - using `Generative.py` result to estimate testing data

- hw2_generative.sh
    - usage: `bash hw2_generative.sh a b train_X train_Y test_X predict.csv`

- hw2_best.py
    - same as `hw2_logistic.py`

- hw2_best.sh
    - usage: `bash hw2_best.sh a b train_X train_Y test_X predict.csv`

- Final Score
    - public score: 
        - Logistic model: `0.85810`
        - Generative model: `0.84557`
        - Strong Baseline: `0.85773`
        - Simple Baseline: `0.84348`
    - private score:
        - Logistic model without regularization: `0.84891`
        - Generative model: `0.84191`
        - Strong Baseline: `0.85382`
        - Simple Baseline: `0.83245`

