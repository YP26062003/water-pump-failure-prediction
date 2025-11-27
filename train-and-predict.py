from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from GroupInfrequentCategories import GroupInfrequentCategories
from SparseToDense import SparseToDenseTransformer
from data_cleaning import *
import os
import argparse
import time
import pandas as pd

# Define the argument parser
parser = argparse.ArgumentParser(description="Train a model and make predictions.")
parser.add_argument('train_input_file', type=str, help="Path to the training input features file (CSV format)")
parser.add_argument('train_labels_file', type=str, help="Path to the training labels file (CSV format)")
parser.add_argument('test_input_file', type=str, help="Path to the test input features file (CSV format)")
parser.add_argument('numerical_preprocessing', type=str, choices=['None', 'StandardScaler'], help="Scaling method for numerical features")
parser.add_argument('categorical_preprocessing', type=str, choices=['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder'], help="Encoding method for categorical features")
parser.add_argument('model_type', type=str, choices=['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'MLPClassifier'], help="Model type to train")
parser.add_argument('test_prediction_output_file', type=str, help="Path to save the test predictions (CSV format)")

# Parse the arguments
args = parser.parse_args()

# Load data
data_dir = "data/"
train_competition_input = pd.read_csv(f"{data_dir}/{args.train_input_file}")
train_competition_labels = pd.read_csv(f"{data_dir}/{args.train_labels_file}")
test_competition_input = pd.read_csv(f"{data_dir}/{args.test_input_file}")
print("**************************************PART 1**************************************")
print("======================================BEFORE======================================")
print("Shape of train_competition_input:", train_competition_input.shape)
print("Shape of train_competition_labels:", train_competition_labels.shape)
print("Training label cardinality: ", train_competition_labels.value_counts())
print("Number of duplicates in train_competition_labels:", train_competition_labels.duplicated().sum())
print("Number of missing rows in train_competition_input:", train_competition_input.isnull().any(axis=1).sum())
# Define preprocessing steps
numerical_transformer = StandardScaler() if args.numerical_preprocessing == 'StandardScaler' else 'passthrough'

if args.categorical_preprocessing == 'OneHotEncoder':
    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
elif args.categorical_preprocessing == 'OrdinalEncoder':
    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
elif args.categorical_preprocessing == 'TargetEncoder':
    categorical_encoder = TargetEncoder()

# Define the model
if args.model_type == 'LogisticRegression':
    model = LogisticRegression()
elif args.model_type == 'RandomForestClassifier':
    model = RandomForestClassifier(n_jobs=-1)
elif args.model_type == 'GradientBoostingClassifier':
    model = GradientBoostingClassifier()
elif args.model_type == 'HistGradientBoostingClassifier':
    model = HistGradientBoostingClassifier()
elif args.model_type == 'MLPClassifier':
    model = MLPClassifier()

# Save the 'id' column before dropping it
test_ids = test_competition_input['id']
# Apply date cleaning
train_competition_input, test_competition_input = handle_dates(train_competition_input, test_competition_input)
train_competition_input, test_competition_input = handle_coordinates(train_competition_input, test_competition_input)
train_competition_input, test_competition_input = drop_columns(train_competition_input, test_competition_input)

# Verify the transformation
print("Test set 'date_recorded':", test_competition_input["date_recorded"].head())

# Select categorical features
df_cat = train_competition_input.select_dtypes(include=["object"], exclude=["int64", "float64"])
df_num = train_competition_input.select_dtypes(include=["float64", "int64"], exclude=["object"])
n_categorical_features = len(df_cat.columns)
n_numerical_features = len(df_num.columns)
print(f"#CATEGORICAL FEATURES: {n_categorical_features}")
print(f"#NUMERICAL features: {n_numerical_features}")

# Find missing categorical columns
missing_cat_columns = find_missing_values(train_competition_input)

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('group_infrequent', GroupInfrequentCategories()),
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
    ('encoder', categorical_encoder),
])

# Preprocessing for categorical features
# categorical_transformer_steps = [
#     ('group_infrequent', GroupInfrequentCategories()),
#     ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
#     ('encoder', categorical_encoder),
# ]
#
# if args.categorical_preprocessing == 'OneHotEncoder':
#     categorical_transformer_steps.append(('svd', TruncatedSVD(n_components=50)))
#
# categorical_transformer = Pipeline(steps=categorical_transformer_steps)

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', numerical_transformer if numerical_transformer != 'passthrough' else 'passthrough')  # Impute missing values
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, df_cat.columns),
        ('num', numerical_transformer, df_num.columns)
    ],
    remainder='passthrough',  # Drop any columns not explicitly transformed
    sparse_threshold=1.0  # Force sparse output
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('sparse_to_dense', SparseToDenseTransformer(model)),
    ('model', model)
])

# Ensure train_competition_labels has only one column
train_competition_labels = train_competition_labels.iloc[:, 1]  # The second column is the target variable

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
start_time_cv = time.time()
scores = cross_val_score(pipeline, train_competition_input, train_competition_labels.values.ravel(), cv=kf, n_jobs=-1)
end_time_cv = time.time()
cv_duration = end_time_cv - start_time_cv

# Print the cross-validation scores
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# Fit the pipeline on the training data
pipeline.fit(train_competition_input, train_competition_labels.values.ravel())

print("======================================AFTER======================================\n")

# Show performance metrics
print("Performance metrics on the training set:")
train_predictions = pipeline.predict(train_competition_input)
train_accuracy = (train_predictions == train_competition_labels.values.ravel()).mean()
print(f"Training set accuracy: {train_accuracy:.4f}")

start_time_train = time.time()
# Make predictions on the test set
test_predictions = pipeline.predict(test_competition_input)
end_time_train = time.time()
train_duration = end_time_train - start_time_train

# Save the predictions to the output file
output = pd.DataFrame({'id': test_ids, 'status_group': test_predictions})
output.to_csv(f"outputs/{args.test_prediction_output_file}", index=False)

# Log the accuracies to a text file
accuracy_dir = "accuracies"
os.makedirs(accuracy_dir, exist_ok=True)
accuracy_file = os.path.join(accuracy_dir, f"{os.path.splitext(args.test_prediction_output_file)[0]}.txt")

with open(accuracy_file, 'w') as f:
    f.write(f"########### {args.model_type}: {args.numerical_preprocessing} + {args.categorical_preprocessing} ###########\n")
    f.write(f"Cross-validation scores: {scores}\n")
    f.write(f"Mean cross-validation score: {scores.mean()}\n")
    f.write(f"Training set accuracy: {train_accuracy:.4f}\n")
    f.write(f"Duration of 5-fold cross-validation: {cv_duration:.2f} seconds\n")
    f.write(f"Duration of training on the whole data: {train_duration:.2f} seconds\n")

print(f"Predictions saved to outputs/{args.test_prediction_output_file}")

