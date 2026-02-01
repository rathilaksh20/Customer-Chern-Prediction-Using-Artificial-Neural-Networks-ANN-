## Customer Churn Prediction Using Artificial Neural Networks (ANN)

Project Description: This project focuses on predicting customer churn for a retail banking dataset using Machine Learning and Deep Learning techniques. The goal was to identify customers likely to exit the bank and understand how customer attributes impact retention.

Key steps covered in this project:
1. Performed data cleaning, exploratory analysis, and feature engineering on a real-world dataset
2. Encoded categorical variables and applied feature scaling using StandardScaler
3. Built and trained an Artificial Neural Network (ANN) using TensorFlow/Keras
4. Evaluated model performance using accuracy and validation metrics
5. Visualized training and validation loss/accuracy to monitor model learning behavior

Model Performance:
1. Achieved ~86% accuracy on the test dataset
2. Used binary classification with sigmoid activation and binary cross-entropy loss

Libraries used in this Project:
1. Python, Pandas, NumPy
2. Scikit-learn (preprocessing, evaluation)
3. TensorFlow & Keras
4. Matplotlib for visualization

Model Performance and Evaluation:
The Artificial Neural Network was trained and evaluated using both training and validation datasets to monitor learning behaviour and generalization performance.
Loss Curve:  
The training and validation loss consistently decrease across epochs, indicating stable convergence and effective optimization without significant overfitting.
![Model Loss (Training vs Validation)](Model%20Loss(Training%20vs%20Validation).png)

Accuracy Curve:  
The model achieves strong and consistent accuracy on both training and validation sets, demonstrating reliable predictive performance on unseen data.
![Model Accuracy (Training vs Validation)](Model%20Accuracy(Training%20vs%20Validation).png)
