PCOS Prediction using Logistic Regression
Description
This project implements a logistic regression model to predict the likelihood of Polycystic Ovary Syndrome (PCOS) based on input features from a dataset. The model is trained using gradient descent and evaluated on a test dataset. There is also a website for it but the connection with flask currently isn't working.

Features
Reads and processes a dataset (pcos_dataset.csv)
Normalizes input features for better training
Implements logistic regression with gradient descent
Computes the cost function and updates weights
Evaluates model performance using accuracy and confusion matrix
Installation
Requirements
Ensure you have Python installed along with the required libraries. Install dependencies using:

bash
Copy
Edit
pip install numpy scikit-learn
Dataset
Place pcos_dataset.csv in the project directory.

Usage
Run the script to train the model and evaluate performance:

bash
Copy
Edit
python main.py
Model Training
The dataset is split into 80% training and 20% testing.
The logistic regression model uses gradient descent with a learning rate of 0.1 and 10,000 iterations.
Normalization is applied to input features before training.
Prediction & Evaluation
The trained model predicts PCOS cases.
Accuracy scores for training and testing data are displayed.
Trained weights are saved in Weights.csv.
Example Output
yaml
Copy
Edit
Training Accuracy: 0.85
Testing Accuracy: 0.82
