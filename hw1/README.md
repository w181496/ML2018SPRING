# Homework 1 - PM2.5 Prediction

- Homework1 web page
    - https://ntumlta2018.github.io/ml-web-hw1/

- train.py
    - using linear regression (Gradient Descent + Adagrad)
    - 12 months，every months have 471 data (10 hours)
    - feature selection
        - [2, 4, 5, 8, 9, 10]
        - CO
        - NO
        - NO2
        - PM10
        - PM2.5
        - RAINFALL
    - Learning rate: 10
    - Iteration: 700000

- model.npy
    - saving the result of `train.py`

- hw1.py
    - using `train.py` result to estimate testing data (test.csv)
    - Linear Regression Only

- hw1.sh
    - usage: `bash hw1.sh test.csv output.csv`

- hw1_best.py
    - my implementation same as `hw1.py`

- hw1_best.sh
    - usage: `bash hw1_best.sh test.csv output.csv`

