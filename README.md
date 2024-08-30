# Student Performance Prediction using ANN

This project demonstrates predicting student performance using Artificial Neural Networks (ANNs). The project uses synthetic data to train and evaluate a neural network model that predicts exam scores based on features like study time, previous grades, and other factors.
In the realm of education, predicting student performance plays a pivotal role in identifying at-risk students early and improving academic outcomes. This project aims to predict student exam scores using Artificial Neural Networks (ANNs), leveraging input features such as study time, previous academic marks, and various socio-economic factors. By analyzing these factors, we seek to develop a predictive model that can effectively forecast students performance. 

## Overview

The project involves:
1. Generating a synthetic dataset with features related to student performance.
2. Building and training an Artificial Neural Network (ANN) to predict exam scores.
3. Evaluating model performance using metrics such as Mean Squared Error (MSE) and R-squared.
4. Classifying student performance based on a threshold and computing accuracy.
5. Visualizing results with scatter plots and loss curves.

## Requirements

Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` (includes Keras)

You can install these packages using pip:

* pip install numpy pandas matplotlib scikit-learn tensorflow

## Usage

### Data Generation and Preprocessing

The script generates a synthetic dataset with the following features:
- **StudyTime**: Random study hours between 1 and 10.
- **PreviousGrades**: Random grades between 0 and 100.
- **OtherFactors**: Random factors from a standard normal distribution.

The target variable, **ExamScore**, is computed with added noise to simulate realistic scenarios.

### Model Building

A Sequential Neural Network model is constructed with:
- **Input Layer**: 64 neurons with ReLU activation.
- **Hidden Layer**: 32 neurons with ReLU activation.
- **Output Layer**: 1 neuron (for regression).

### Training and Evaluation

The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

It is trained for 50 epochs with a batch size of 32. Model performance is evaluated on a test set using MSE and R-squared metrics.

### Classification and Accuracy

The model's predictions are classified based on a threshold (e.g., a score of 60) to determine pass/fail status. Accuracy is calculated by comparing predicted classifications to actual classifications.

### Visualization

The script includes plots for:
- **Actual vs. Predicted Exam Scores**
- **Training and Validation Loss** over epochs

### Output
![image](https://github.com/user-attachments/assets/e396fd4e-bee7-470b-8804-0ca81937e759)
![image](https://github.com/user-attachments/assets/2f5dacad-e9ee-4bf7-82ab-41077f553bb2)



