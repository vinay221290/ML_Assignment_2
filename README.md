# Classification Models – Machine Learning Project

## 1. Project Overview

This project is developed as part of a Machine Learning assignment. It provides an interactive web application built using Streamlit to evaluate and compare five classification models.

Users can upload a test dataset in CSV format, select a classification model, and evaluate its performance using different classification metrics.

## 2. Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib / Pickle (for saving and loading trained models)

## 3. Project Structure

```text
classification-models/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── saved_models/
│   ├── model_1.pkl
│   ├── model_2.pkl
│   ├── model_3.pkl
│   ├── model_4.pkl
│   └── model_5.pkl
│
└── notebooks/
    └── model_training.ipynb
```

*Note: Update filenames and model names to match your actual project.*

## 4. Machine Learning Models

The application supports five classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## 5. Application Features

- Upload test datasets in CSV format.
- Select from five trained classification models.
- Load previously saved machine learning models.
- Generate predictions using the selected model.
- Evaluate model performance using six evaluation metrics.
- Compare classification model performance through the Streamlit interface.

## 6. Evaluation Metrics

The application evaluates model performance using:

1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. Confusion Matrix
6. Classification Report

## 7. Installation and Setup

Clone the repository:

```bash
git clone
https://mlassignment2-hud2hvupbpjcubqthdiefw.streamlit.app/

```

Navigate to the project directory:

```bash
cd classification-models
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 8. Run the Application

Execute the following command:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 9. How to Use

1. Launch the Streamlit application.
2. Upload the test dataset (`test_data.csv`).
3. Select one of the five classification models.
4. Run the model evaluation.
5. Review the generated predictions and evaluation metrics.

## 10. Saved Models

All five trained classification models are stored in the `saved_models` directory.

These pre-trained models are loaded by the Streamlit application to generate predictions without requiring model training every time the application starts.

## 11. Author

Machine Learning Assignment Project

**Project:** Classification Models  
**Language:** Python  
**Interface:** Streamlit
