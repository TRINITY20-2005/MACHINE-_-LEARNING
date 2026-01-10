# MACHINE-_-LEARNING
THis inlcudes basic NL concepts and a series where i will upload topics as i learn thorugh them.
Machine Learning & AI Implementation Suite
This repository contains a collection of Python implementations covering the full spectrum of modern data science. It ranges from foundational statistical models to advanced neural network architectures and recommendation systems.

📂 Project Overview
1. Dimensionality Reduction & Visualization
Script: PCA.py

Focus: Principal Component Analysis (PCA).

Description: Implements PCA on the Iris dataset to reduce 4D features into 2D principal components for visualization, while calculating the Explained Variance Ratio to ensure data integrity.

2. Predictive Modeling (Supervised Learning)
Linear Regression (linear_regression.py): A regression model predicting housing area based on price, evaluating performance using Mean Squared Error (MSE) and R² Score.

Logistic Regression (LOGISTIC_REGRESSION.py): A classification tool for student success. It features a dynamic visualization engine that draws Decision Boundaries for multi-feature inputs or Sigmoid Curves for single-feature analysis.

3. Large Language Models (LLM) Architecture
Script: pre.py

Focus: Transformer Architecture (GPT-2).

Description: Demonstrates the structural initialization of a GPT-2 model using the Hugging Face transformers library. It defines the configuration for 124 million parameters, including embedding dimensions, layers, and attention heads.

4.This script provides a foundation for predicting housing prices using a Support Vector Regressor (SVR). It focuses on the critical data preprocessing stage, transforming raw housing data into a format suitable for high-performance machine learning.

        Key Features
          Automated Categorical Encoding: Utilizes One-Hot Encoding via pandas to transform qualitative features (like furnishing status or presence of air conditioning) into binary numerical values.

          Multicollinearity Prevention: Employs drop_first=True during dummy variable creation to avoid the dummy variable trap, ensuring model stability.

              Data Partitioning: Implements a standard 80/20 train-test split to allow for robust validation and performance testing.

                  Feature Engineering Readiness: Sets the stage for feature scaling and SVR implementation using the scikit-learn ecosystem.

                    Technologies Used
                Python: Core programming language.

                 Pandas: Data manipulation and cleaning.

                      Scikit-Learn: Machine learning utilities, including data splitting and model metrics.

                      How it Works
                        Load: Reads the Housing.csv dataset into a DataFrame.

                     Process: Identifies categorical columns and converts them to numerical format.

                       Separate: Isolates the target variable (price) from the input features.

                        Split: Divides the data into training and testing subsets using a fixed random_state for reproducibility.
  5.LoRA Fine-Tuning with PEFT 🚀
This repository demonstrates how to implement Low-Rank Adaptation (LoRA) to efficiently fine-tune Large Language Models (LLMs) using the peft (Parameter-Efficient Fine-Tuning) library.

The provided script applies LoRA to DistilGPT2, reducing the number of trainable parameters to approximately 0.36% of the original model size while maintaining the ability to adapt to specific tasks.

💡 Key Features
Parameter Efficiency: Train only a small fraction (294,912) of the total parameters (82M+), significantly reducing memory usage and storage requirements.

Targeted Adaptation: Specifically targets the c_attn modules (attention layers) of the GPT architecture for optimization.

Hugging Face Integration: Built using the transformers and peft ecosystems for seamless model loading and configuration.
🛠️ Tech Stack
Languages: Python

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Transformers (Hugging Face)

Techniques: Data Standardization, Train/Test Splitting, Feature Scaling, Vector Similarity, Neural Network Configuration.
