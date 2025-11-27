from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from GroupInfrequentCategories import GroupInfrequentCategories
from SparseToDense import SparseToDenseTransformer
from data_cleaning import *
import os
import optuna
import optuna.visualization as vis
import pandas as pd
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Train a model and make predictions.")
parser.add_argument('train_input_file', type=str, help="Path to the training input features file (CSV format)")
parser.add_argument('train_labels_file', type=str, help="Path to the training labels file (CSV format)")
parser.add_argument('test_input_file', type=str, help="Path to the test input features file (CSV format)")

# Parse the arguments
args = parser.parse_args()
# Load data
data_dir = "data/"
train_competition_input = pd.read_csv(f"{data_dir}/{args.train_input_file}")
train_competition_labels = pd.read_csv(f"{data_dir}/{args.train_labels_file}")
test_competition_input = pd.read_csv(f"{data_dir}/{args.test_input_file}")
train_competition_labels = train_competition_labels.iloc[:, 1]

# Save the 'id' column before dropping it
test_ids = test_competition_input['id']
# Apply date cleaning
train_competition_input, test_competition_input = handle_dates(train_competition_input, test_competition_input)
train_competition_input, test_competition_input = handle_coordinates(train_competition_input, test_competition_input)
train_competition_input, test_competition_input = drop_columns(train_competition_input, test_competition_input)

###################### Part 2 ##############################
print("**************************************PART 2**************************************")
def define_config_space(trial):
    # Preprocessing choices
    categorical_encoder = trial.suggest_categorical(
        'categorical_encoder', ['OneHotEncoder', 'OrdinalEncoder', 'TargetEncoder']
    )
    numerical_scaler = trial.suggest_categorical(
        'numerical_scaler', ['StandardScaler', 'None']
    )
    imputation_strategy_num = trial.suggest_categorical(
        'numerical_imputation_strategy', ['mean', 'median', 'most_frequent']
    )
    imputation_strategy_cat = 'most_frequent'  # Use most_frequent for categorical data
    group_infrequent_threshold = trial.suggest_int(
        'group_infrequent_threshold', 50, 200
    )
    cat_threshold = trial.suggest_int(
        'cat_threshold', 1, 10
    )

    reduction_size = trial.suggest_int(
        'svd_n_components', 50, 100
    )

    # Model choices
    model_name = trial.suggest_categorical(
        'model', ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier', 'HistGradientBoostingClassifier', 'MLPClassifier']
    )

    # Define preprocessing steps
    categorical_transformer = Pipeline(steps=[
        ('group_infrequent', GroupInfrequentCategories(
            threshold=group_infrequent_threshold, cat_threshold=cat_threshold)),
        ('imputer', SimpleImputer(strategy=imputation_strategy_cat)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True) if categorical_encoder == 'OneHotEncoder'
        else OrdinalEncoder(handle_unknown='use_encoded_value',
                            unknown_value=-1) if categorical_encoder == 'OrdinalEncoder'
        else TargetEncoder())
    ])

    # Define preprocessing steps
    # categorical_transformer_steps = [
    #     ('group_infrequent', GroupInfrequentCategories(
    #         threshold=group_infrequent_threshold, cat_threshold=cat_threshold)),
    #     ('imputer', SimpleImputer(strategy=imputation_strategy_cat)),
    #     (
    #     'encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False) if categorical_encoder == 'OneHotEncoder'
    #     else OrdinalEncoder(handle_unknown='use_encoded_value',
    #                         unknown_value=-1) if categorical_encoder == 'OrdinalEncoder'
    #     else TargetEncoder())
    # ]
    #
    # if categorical_encoder == 'OneHotEncoder':
    #     categorical_transformer_steps.append(
    #         ('svd', TruncatedSVD(n_components=reduction_size)))
    #
    # categorical_transformer = Pipeline(steps=categorical_transformer_steps)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy_num)),
        ('scaler', StandardScaler() if numerical_scaler == 'standard' else 'passthrough')
    ])

    df_cat = train_competition_input.select_dtypes(include=["object"], exclude=["int64", "float64"])
    df_num = train_competition_input.select_dtypes(include=["float64", "int64"], exclude=["object"])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, df_cat.columns),
            ('num', numerical_transformer, df_num.columns)
        ],
        remainder='passthrough',
        sparse_threshold=1.0 # Force the sparse output
    )

    # Define model
    if model_name == 'LogisticRegression':
        model = LogisticRegression(
            C=trial.suggest_float('logistic_regression_C', 1e-4, 1e4, log=True),
            penalty=trial.suggest_categorical('logistic_regression_penalty', ['l1', 'l2']),
            solver='liblinear',
            max_iter=trial.suggest_int('logistic_regression_max_iter', 100, 1000)
        )
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('random_forest_n_estimators', 50, 500),
            max_depth=trial.suggest_int('random_forest_max_depth', 3, 15),
            min_samples_split=trial.suggest_int('random_forest_min_samples_split', 2, 10),
            n_jobs=-1
        )
    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('gradient_boosting_n_estimators', 50, 500),
            learning_rate=trial.suggest_float('gradient_boosting_learning_rate', 1e-3, 1e-1, log=True),
            max_depth=trial.suggest_int('gradient_boosting_max_depth', 3, 15)
        )
    elif model_name == 'HistGradientBoostingClassifier':
        model = HistGradientBoostingClassifier(
            max_iter=trial.suggest_int('hist_gradient_boosting_max_iter', 50, 500),
            learning_rate=trial.suggest_float('hist_gradient_boosting_learning_rate', 1e-3, 1e-1, log=True),
            max_depth=trial.suggest_int('hist_gradient_boosting_max_depth', 3, 15)
        )
    elif model_name == 'MLPClassifier':
        model = MLPClassifier(
            hidden_layer_sizes=(trial.suggest_int('mlp_hidden_layer_size', 50, 200),),
            alpha=trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float('mlp_learning_rate_init', 1e-4, 1e-2, log=True),
            max_iter=trial.suggest_int('mlp_max_iter', 100, 1000)
        )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('sparse_to_dense', SparseToDenseTransformer(model)),
        ('model', model)
    ])

    return pipeline

def objective(trial):
    # Define the pipeline using the configuration space
    pipeline = define_config_space(trial)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Evaluate the pipeline using cross-validation
    scores = cross_val_score(pipeline, train_competition_input, train_competition_labels.values.ravel(), cv=kf, scoring='accuracy', n_jobs=-1)
    # Return the mean accuracy
    return scores.mean()

if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)  # Maximize accuracy
    study.optimize(objective, n_trials=50)  # Run 50 trials

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Plot optimization history
    vis.plot_optimization_history(study)
    # Plot parameter importance
    vis.plot_param_importances(study)
    # Plot parallel coordinate plot
    vis.plot_parallel_coordinate(study)

    # Retrain the model on the full training dataset using the best hyper-parameters
    best_pipeline = define_config_space(trial)
    best_pipeline.fit(train_competition_input, train_competition_labels.values.ravel())

    # Evaluate on the training set
    train_accuracy = best_pipeline.score(train_competition_input, train_competition_labels.values.ravel())
    print(f"Training Set Accuracy: {train_accuracy}")

    # Log the accuracies to a text file
    accuracy_dir = "accuracies_after_hpo"
    os.makedirs(accuracy_dir, exist_ok=True)
    accuracy_file = os.path.join(accuracy_dir, "best_model_accuracy.txt")

    with open(accuracy_file, 'w') as f:
        f.write("Best trial:\n")
        f.write(f"  Value (accuracy): {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

        f.write("########### Best Model After HPO ###########\n")
        f.write(f"Training Set Accuracy: {train_accuracy:.4f}\n")
