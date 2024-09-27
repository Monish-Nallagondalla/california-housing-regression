# california-housing-regression
A Python implementation of Multiple Linear Regression using the California Housing Dataset to predict house prices. The project demonstrates data preprocessing, model training, evaluation metrics, and saving the trained model using pickle


# Multiple Linear Regression with California Housing Dataset

This project demonstrates how to implement **Multiple Linear Regression** using the **California Housing dataset**. The objective is to predict house prices based on various features like the population, number of rooms, and income.

## Project Overview

The project covers:
- Data loading and preprocessing
- Multiple Linear Regression model training
- Model evaluation using R-squared, Mean Squared Error, and Mean Absolute Error
- Visualizing model performance and residuals
- Saving the trained model using `pickle`

## Libraries and Dependencies

The following libraries are required to run the code:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

To install these dependencies, run:

```bash
pip install -r requirements.txt

```

Project Structure
regression.ipynb: The Jupyter notebook containing the full implementation of the project.
regression.pkl: The saved Linear Regression model using pickle.
README.md: This file, which provides an overview of the project.
requirements.txt: A file listing the required libraries for the project.
Model Evaluation
The following metrics were used to evaluate the model performance:

R-squared: Coefficient of determination, used to evaluate how well the model fits the data.
Mean Squared Error (MSE): Average of the squares of the errors, providing insight into the error magnitude.
Mean Absolute Error (MAE): The average of the absolute differences between the predicted and actual values.
Root Mean Squared Error (RMSE): The square root of the MSE, giving an intuitive sense of the error magnitude.


Visualizations
The following visualizations were created to check the assumptions of the regression model:

Scatter plot between actual and predicted values
Distribution plot of residuals to check for normality
Residual plot to identify patterns in residuals
Usage
Once you have cloned the repository, you can load the saved model and use it to make predictions as follows:

import pickle

# Load the model
model = pickle.load(open('regression.pkl', 'rb'))

# Example prediction
model.predict([[0.75, -1.31, -0.39, 0.12, 0.12, -0.68, 0.19, 0.19]])
