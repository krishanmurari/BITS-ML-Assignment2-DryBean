# BITS-ML-Assignment2-DryBean

## 1. Problem statement

This project addresses the classification of dry bean varieties using their geometric and morphological features. The objective is to train and compare several machine learning models to identify the most accurate classifier for bean type prediction based on numeric image-derived measurements.

The problem is a supervised multi-class classification task where the model learns from 16 input features and predicts one of seven bean classes.

## 2. Dataset description

The dataset used in this project is the Dry Bean dataset from the UCI Machine Learning Repository.

- Source: UCI Machine Learning Repository
- Total rows: 13,611
- Number of numerical features: 16
- Number of target classes: 7 bean classes
- Target variable: `Class`

The dataset contains shape and size descriptors extracted from bean images, including features such as area, perimeter, major/minor axis lengths, shape factors, compactness, aspect ratio, and eccentricity.

## 3. GitHub repository link

Repository: https://github.com/krishanmurari/BITS-ML-Assignment2-DryBean

## 4. Model comparison table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| SVM | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

## 5. Observations for all five models

### Logistic Regression
- Achieved strong overall performance with high accuracy and AUC.
- Good balance between precision and recall, indicating stable classification across bean classes.
- Suitable as a fast and interpretable benchmark model.

### Decision Tree
- Performed the weakest among all models.
- Lower AUC and MCC indicate weaker discrimination across classes compared to the other classifiers.
- Simpler and more interpretable but less robust for this dataset.

### KNN
- Delivered competitive performance, with accuracy close to the top models.
- Slightly below Logistic Regression and SVM in AUC and F1.
- Effective for local pattern recognition but may be less scalable for larger datasets.

### Gaussian Naive Bayes
- Showed decent performance with a very high AUC, suggesting good ranking ability.
- Accuracy and MCC were lower than Logistic Regression and SVM.
- Works well as a lightweight probabilistic baseline but is less precise than stronger classifiers.

### SVM
- Produced the best results overall across the evaluation metrics.
- Highest accuracy, AUC, F1, and MCC among all models.
- Best balance of precision and recall, making it the strongest choice for this classification task.

## 6. Overall winner

The overall winner is the Support Vector Machine (SVM) model.

It achieved the highest:
- Accuracy: 0.9221
- AUC: 0.9937
- F1 Score: 0.9222
- MCC: 0.9059

This makes SVM the best-performing classifier in this project based on the reported evaluation metrics.

## 7. Instructions to run the application

### Prerequisites

- Python 3.10+
- pip

### Clone the repository

```bash
git clone https://github.com/krishanmurari/BITS-ML-Assignment2-DryBean.git
cd BITS-ML-Assignment2-DryBean
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

### Use the app

1. Select the model from the sidebar.
2. Upload a CSV file containing the same feature columns used during training.
3. View predictions for the uploaded dataset.
4. If the CSV includes the `Class` column, the app will display evaluation metrics, confusion matrix, and classification report.
5. Download the prediction results as a CSV file.

## Additional notes

- The feature names are stored in `models/feature_names.json` to preserve the same input order used while training.
- The trained model files are stored in the `models/` directory and are loaded by the Streamlit dashboard.
- The project is intended for academic use and demonstrates a complete machine learning workflow from data collection to model evaluation.
