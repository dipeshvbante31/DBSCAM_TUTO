# DBSCAN + EDA + Machine Learning + Prediction

A beginner-friendly Jupyter Notebook project combining **Exploratory Data Analysis (EDA)**, **DBSCAN clustering**, and a **Random Forest** model for classification or regression.

The notebook includes built-in sample data, so you can run it immediately without downloading a dataset.

## Features

- Load a CSV dataset or use built-in sample data
- Inspect dataset shape, columns, data types, and summary statistics
- Check and remove duplicate rows
- Analyze missing values
- Visualize target and numerical feature distributions
- Detect potential outliers with boxplots
- Explore numerical correlations with a heatmap
- Preprocess numerical and categorical features
- Apply DBSCAN clustering and inspect cluster counts
- Use a k-distance plot to explore DBSCAN's `eps` parameter
- Visualize clusters with PCA when possible
- Calculate a silhouette score when supported by the clustering result
- Automatically select Random Forest classification or regression
- Evaluate the supervised model and display feature importance
- Make a sample prediction
- Save the trained model and test predictions

## Project Structure

```text
DBSCAN-EDA-ML/
├── DBSCAN_EDA_ML_Beginner.ipynb
├── data/
│   └── dataset.csv              # Optional: your own dataset
└── outputs/
    ├── trained_random_forest_pipeline.pkl  # Created by notebook
    └── predictions.csv                     # Created by notebook
```

The `data/` folder and CSV are optional when using built-in sample data. The `outputs/` folder is created automatically when the notebook reaches the save step.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or Jupyter support in VS Code
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

## How to Run

### Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `DBSCAN_EDA_ML_Beginner.ipynb`.
3. Select **Runtime → Run all**.
4. The notebook uses built-in sample data by default, so no CSV upload is needed.

### Option 2: VS Code

1. Install Python and VS Code.
2. Install the **Python** and **Jupyter** extensions in VS Code.
3. Open the notebook file.
4. Select a Python kernel when prompted.
5. Click **Run All** at the top of the notebook.

If a package is missing, run this in a notebook cell:

```python
%pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

Restart the kernel if prompted, then choose **Run All** again.

### Option 3: Jupyter Notebook from a terminal

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open `DBSCAN_EDA_ML_Beginner.ipynb` in the browser and select **Run → Run All Cells**.

## Use Your Own Dataset

The notebook starts with sample data so that it works out of the box.

1. Put your CSV in the project folder (for example, `data/dataset.csv`).
2. Open the **Configuration** cell near the top of the notebook.
3. Change the settings:

```python
USE_SAMPLE_DATA = False
DATA_PATH = "data/dataset.csv"
TARGET_COLUMN = "target"
```

4. Replace `"target"` with the exact name of the column you want to predict.
5. Run the notebook from the beginning.

The target column is used for supervised learning and excluded from DBSCAN's input features. The notebook expects tabular CSV data and a usable target column.

## Workflow

1. **Load data:** use built-in sample data or read a CSV.
2. **Inspect data:** review columns, types, summary statistics, and duplicates.
3. **EDA:** examine missing values, target distribution, numerical distributions, outliers, and correlations.
4. **Preprocess for DBSCAN:** impute missing values, scale numeric features, and one-hot encode categorical features.
5. **Cluster:** fit DBSCAN and inspect cluster labels and counts.
6. **Visualize:** use PCA to display clusters in two dimensions when possible.
7. **Train:** split the data and train a Random Forest classifier or regressor.
8. **Evaluate:** report classification metrics or regression errors.
9. **Interpret:** inspect feature importance.
10. **Predict and save:** make an example prediction and save the model and test predictions.

## Output Files

After the notebook completes its save step, it creates:

- `outputs/trained_random_forest_pipeline.pkl` — fitted preprocessing and Random Forest pipeline.
- `outputs/predictions.csv` — actual and predicted values for the held-out test set.

The saved pipeline includes preprocessing. New data should have the same original feature columns used during training.

## Notes and Limitations

- DBSCAN is sensitive to `eps`, `min_samples`, feature scaling, and dataset characteristics. The k-distance plot is a guide, not a guarantee of ideal parameters.
- DBSCAN label `-1` represents noise.
- A silhouette score is only calculated when at least two non-noise clusters are present.
- The notebook uses a simple automatic rule to choose classification or regression. For unusual targets, review the `is_classification` setting in the notebook.
- Feature importance describes how the fitted Random Forest uses features; it does not establish causation.
- The example prediction uses the first row of the dataset. Replace it with your own input values for a real use case.
- The built-in sample data demonstrates the workflow and is not intended for real-world conclusions.

## License

No license is specified. Add a license file if you intend to publish or distribute this project.
