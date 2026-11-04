# BITS-ML-Assignment2-DryBean

This project implements a dry bean classification workflow for the BITS WILP M.Tech AIML assignment. It uses multiple machine learning classifiers to predict bean types from morphological features and includes a Streamlit web app for evaluating the trained models on uploaded CSV datasets.

## Project Overview

The task is based on the Dry Bean Dataset from the UCI Machine Learning Repository. The dataset contains geometric and shape-based features extracted from images of beans, and the target variable is the bean class.

The notebook trains and compares several classifiers, including:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

The best-performing model is saved in the `models/` directory and can be loaded by the app for predictions and evaluation.

## Repository Structure

- `BITS_ML_Assignment2.ipynb` — training notebook with data loading, preprocessing, model training, and evaluation
- `app.py` — Streamlit application for model inference and performance assessment
- `test_data.csv` — sample test dataset containing features and labels
- `models/` — saved trained models and feature metadata
- `requirements.txt` — Python dependencies for the project

## Dataset

The project uses the Dry Bean dataset from UCI (`ucimlrepo`), which includes features such as:
- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRatio
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- Roundness
- Compactness
- ShapeFactor1-4

The target column is `Class`, which contains the bean category labels.

## Setup

1. Clone the repository.
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Notebook

Open the Jupyter notebook:

```bash
jupyter notebook BITS_ML_Assignment2.ipynb
```

This notebook performs:
- dataset download and preprocessing
- train/test splitting
- model training
- saved model generation
- evaluation metrics and confusion matrix analysis

## Run the Streamlit App

Start the application with:

```bash
streamlit run app.py
```

Then in the browser:
- select a trained model from the sidebar,
- upload a CSV file containing the required features,
- view predictions,
- download the prediction output,
- inspect evaluation metrics if the uploaded file includes the `Class` column.

## Notes

- The app expects the uploaded CSV to contain the same feature columns as the training data.
- Feature names are stored in `models/feature_names.json` to keep the input order consistent.
- Model files in `models/` are saved using `joblib` for reuse in the app.

## Requirements

The project depends on:
- Python 3.10+
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- ucimlrepo

## License

This project is intended for academic coursework and learning purposes.
