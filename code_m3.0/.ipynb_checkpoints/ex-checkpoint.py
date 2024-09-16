import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor









# Load the data
file_path = "data_ra_norm_filled_all.xlsx"
data = pd.read_excel(file_path)

# Define columns
data_columns = [
    'Wetland Type - Provincial Class',
    'Wetland Type - Federal Class',
    'Water Regime Indicator',
    'Specific Vegetation Type',
    '% Vegetation Cover for Specific Vegetation Cover Types',
    '% High Woody Canopy Cover (>5m)',
    'Phragmites present (Y/N)',
    'Soil Type',
    '% of Surface Water Present',
    'Depth of Saturation (cm)',
    'Average Depth of Living Moss (cm)',
    'Average Total Depth of Organics',
    'Average Organic Depth (cm)',
    'Hydrogeomorphic Class',
    '% Moss Cover'
]

results_columns = ['NR']

# Prepare data for regression
X = data[data_columns]
y = data[results_columns[0]]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models 
# RandomForestRegressor(max_depth=100),
models = [
    Ridge(), DecisionTreeRegressor(), RandomForestRegressor(max_depth=20),GradientBoostingRegressor(),
    AdaBoostRegressor(), KNeighborsRegressor(), MLPRegressor(max_iter=1000),ElasticNet(max_iter=1000),SGDRegressor(max_iter=1000),SVR(cache_size=1000),BayesianRidge(max_iter=1000),KernelRidge(),LinearRegression(), RANSACRegressor(), TheilSenRegressor()
]

# Define hyperparameters to search for each model

param_grid = {
    'Ridge': {
        'ridge__alpha': [0.1, 0.5, 1.0],
        'ridge__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    },
    'DecisionTreeRegressor': {
        'decisiontreeregressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'decisiontreeregressor__splitter': ['best', 'random'],
        'decisiontreeregressor__min_samples_split': [1, 2, 3, 4, 5],
        'decisiontreeregressor__max_features': [0, 1, 2, 3, 'sqrt', 'log2']
    },
    'RandomForestRegressor': {
        'randomforestregressor__n_estimators': [1, 20, 50, 100, 200, 500],
        'randomforestregressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'randomforestregressor__min_samples_split': [1, 2, 3, 4, 5, 6],
        'randomforestregressor__max_features': [0, 1, 2, 3, 'sqrt', 'log2'],
        'randomforestregressor__ccp_alpha': [0.0, 0.001, 0.01, 0.1, 1, 10]
    },
    'GradientBoostingRegressor': {
        'gradientboostingregressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'gradientboostingregressor__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'gradientboostingregressor__n_estimators': [1, 20, 50, 100, 200, 500],
        'gradientboostingregressor__subsample': [0.1, 0.25, 0.5, 0.75, 1.0],
        'gradientboostingregressor__criterion': ['squared_error', 'friedman_mse'],
        'gradientboostingregressor__min_samples_split': [2, 4, 8, 10],
        'gradientboostingregressor__max_depth': [1, 2, 3, 4, 5],
        'gradientboostingregressor__max_features': [0, 1, 2, 3, 'sqrt', 'log2'],
        'gradientboostingregressor__alpha': [0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
        'gradientboostingregressor__warm_start': [True, False],
        'gradientboostingregressor__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        'gradientboostingregressor__ccp_alpha': [0.0, 0.001, 0.01, 0.1, 1, 10]
    },
    'AdaBoostRegressor': {
        'adaboostregressor__n_estimators': [1, 20, 50, 100, 200, 500],
        'adaboostregressor__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'adaboostregressor__loss': ['linear', 'square', 'exponential']
    },
    'KNeighborsRegressor': {
        'kneighborsregressor__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25],
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'kneighborsregressor__leaf_size': [5, 10, 15, 20, 30, 50],
        'kneighborsregressor__p': [0.5, 1, 2, 2.5, 5],
        'kneighborsregressor__metric': ['cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean']
    },
    'MLPRegressor': {
        'mlpregressor__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100, 50, 100), (100, 100, 100), (50, 100, 150), (100, 100, 100, 100), (100, 100, 100, 100, 100)],
        'mlpregressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'mlpregressor__solver': ['lbfgs', 'sgd', 'adam'],
        'mlpregressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'mlpregressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'mlpregressor__learning_rate_init': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'mlpregressor__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'mlpregressor__warm_start': [True, False],
        'mlpregressor__nesterovs_momentum': [True, False],
        'mlpregressor__beta_1': [0.75, 0.8, 0.9, 0.95, 0.99],
        'mlpregressor__beta_2': [0.85, 0.9, 0.95, 0.98, 0.99, 0.999],
        'mlpregressor__epsilon': [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.00001, 0.001]
    },
    'ElasticNet': {
        'elasticnet__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'elasticnet__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'elasticnet__fit_intercept': [True, False],
        'elasticnet__precompute': [True, False],
        'elasticnet__copy_X': [True, False],
        'elasticnet__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'elasticnet__warm_start': [True, False],
        'elasticnet__positive': [True, False],
        'elasticnet__selection': ['cyclic', 'random']
    },
    'SGDRegressor': {
        'sgdregressor__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'sgdregressor__penalty': ['l2', 'l1', 'elasticnet', None],
        'sgdregressor__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'sgdregressor__l1_ratio': [0.0, 0.15, 0.25, 0.5, 0.75, 1.0],
        'sgdregressor__fit_intercept': [True, False],
        'sgdregressor__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'sgdregressor__shuffle': [True, False],
        'sgdregressor__epsilon': [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 5.0],
        'sgdregressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'sgdregressor__eta0': [0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 5.0],
        'sgdregressor__power_t': [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0],
        'sgdregressor__early_stopping': [False, True],
        'sgdregressor__n_iter_no_change': [1, 5, 25, 50, 100],
        'sgdregressor__warm_start': [True, False],
        'sgdregressor__average': [True, False, 2, 5, 10]
    },
    'SVR': {
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'svr__degree': [1, 2, 3, 5, 10],
        'svr__gamma': ['scale', 'auto', 1.0, 5.0],
        'svr__coef0': [0.0, 0.5, 1.0],
        'svr__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'svr__C': [0.01, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0],
        'svr__epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
        'svr__shrinking': [True, False]
    },
    'BayesianRidge': {
        'bayesianridge__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'bayesianridge__alpha_1': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'bayesianridge__alpha_2': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'bayesianridge__lambda_1': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'bayesianridge__lambda_2': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'bayesianridge__compute_score': [True, False],
        'bayesianridge__fit_intercept': [True, False],
        'bayesianridge__copy_X': [True, False]
    },
    'KernelRidge': {
        'kernelridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'kernelridge__degree': [1, 2, 3, 5, 10],
        'kernelridge__coef0': [0.0, 0.5, 1.0]
    },
    'LinearRegression': {
        'linearregression__fit_intercept': [True, False],
        'linearregression__copy_X': [True, False],
        'linearregression__positive': [True, False]
    },
    'RANSACRegressor': {
        'ransacregressor__min_samples': [None, 1, 2, 5, 10, 50],
        'ransacregressor__max_trials': [1, 10, 50, 100, 150],
        'ransacregressor__loss': ['absolute_error', 'squared_error']
    },
    'TheilSenRegressor': {
        'theilsenregressor__fit_intercept': [True, False],
        'theilsenregressor__copy_X': [True, False],
        'theilsenregressor__max_subpopulation': [1, 10, 100, 1000, 10000, 100000],
        'theilsenregressor__n_subsamples': [None, 1, 5, 10, 25],
        'theilsenregressor__tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    }
}



# Train and tune hyperparameters for each model
best_models = {}

for model in models + ['TensorFlow']:  # Add TensorFlow model to the loop
    print(model)
    if model == 'TensorFlow':
        # Define the TensorFlow model
        model_tf = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile the TensorFlow model
        model_tf.compile(optimizer='adam', loss='mean_squared_error')

        # Standardize the data for TensorFlow model
        scaler_tf = StandardScaler()
        X_train_scaled_tf = scaler_tf.fit_transform(X_train)
        X_test_scaled_tf = scaler_tf.transform(X_test)

        # Train the TensorFlow model
        model_tf.fit(X_train_scaled_tf, y_train, epochs=100, batch_size=32, validation_split=0.2)

        # Evaluate the TensorFlow model
        y_pred_tf = model_tf.predict(X_test_scaled_tf)
        rmse_tf = mean_squared_error(y_test, y_pred_tf, squared=False)
        print(f"TensorFlow RMSE: {rmse_tf}")

        # Add TensorFlow model to best_models
        best_models['TensorFlow'] = model_tf
    else:
        print("1")
        model_name = model.__class__.__name__
        pipeline = make_pipeline(StandardScaler(), model)
        print("2")
        # Perform grid search for hyperparameters
        if model_name in param_grid:
            print("3")
            grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
            print("4")
            grid_search.fit(X_train, y_train)
            print("5")
            best_models[model_name] = grid_search.best_estimator_
            print("6")
            print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            best_models[model_name] = pipeline

# Make predictions using the best models
for model_name, model in best_models.items():
    print(f"Model: {model_name}")

    # Make predictions
    if model_name == 'TensorFlow':
        y_pred = y_pred_tf  # Use predictions from TensorFlow model
    else:
        y_pred = model.predict(X_test)

    # Calculate and print RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # Show real and predicted results for the first 5 samples
    print("Sample predictions:")
    for i in range(5):
        print(f"Sample {i+1}: Real NR = {y_test.iloc[i]}, Predicted NR = {y_pred[i]}")

    print("\n")

for index, estimator in enumerate(models):
    model_name = type(estimator).__name__
    file_name = f"{model_name}.joblib"
    joblib.dump(estimator, file_name)
    print(f"Model '{model_name}' saved as '{file_name}'")