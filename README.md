#HYBRID DEEP LEARNING FRAUD DETECTION SYSTEM

An end-to-end Machine Learning solution using a hybrid **Unsupervised (SOM)** and **Supervised (ANN)** approach to detect fraudulent credit card applications with **87.54% accuracy**.

## 🚀 Live Demo
[Check out the live app here!](https://share.streamlit.io/) *(Replace this with your link once deployed)*

## 🧠 Project Architecture
This project implements a dual-stage neural network pipeline:
1. **Self-Organizing Map (SOM):** An unsupervised learning stage used to map high-dimensional data and identify topological outliers (potential fraud clusters).
2. **Artificial Neural Network (ANN):** A supervised stage using a Dense architecture with **Binary Cross-Entropy** loss to classify the probability of fraud for each application.

## Performance
- **Validation Accuracy:** 87.54%
- **Final Model Loss:** 0.2106
- **Dataset:** UCI Statlog (Australian Credit Approval)

#Tech Stack
- **Python** (Pandas, Numpy, Scikit-Learn)
- **Deep Learning:** TensorFlow, Keras, MiniSom
- **Deployment:** Streamlit
- **Visualization:** Matplotlib, Seaborn

#  How to Run Locally
1. Clone the repo: 
   ```bash
   git clone [https://github.com/NILAMBARMANDAL/Credit-Card-Fraud-Detection.git](https://github.com/NILAMBARMANDAL/Credit-Card-Fraud-Detection.git)