import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import time

# Read the data from the CSV file
diamonds_data = pd.read_csv("diamonds.csv")

# Features and target variable
features = ["carat", "depth", "table", "length", "width"]
target = "price"

# Results
results = []

# Perform 5 iterations
for i in range(5):
    # Sample 5000 random data points (repeat with 10000, 25000)
    sampled_data = diamonds_data.sample(5000)
    
    # Split data into features and target variable
    X = sampled_data["carat", "depth", "table", "length", "width"]
    y = sampled_data["price"]
    
    # Split data into train and test sets for forecasting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1917)
    
    # Perform iterations for each number of layers (1 to 5)
    for n_layers in range(1, 6):
        # Initialize MLPRegressor with varying number of hidden layers
        mlp_regressor = MLPRegressor(hidden_layer_sizes=(n_layers,), random_state=1917)
        
        # Fit the model and measure fitting time
        start_time = time.time()
        mlp_regressor.fit(X_train, y_train)
        end_time = time.time()
        fitting_time = end_time - start_time
        
        # Predict on the test set
        y_pred = mlp_regressor.predict(X_test)
        
        # Calculate R^2 score and MSE
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Results
        results.append({
            'Iteration': i + 1,
            'Number of Layers': n_layers,
            'R^2 Score': r2,
            'MSE': mse,
            'Fitting Time (seconds)': fitting_time
        })

# Convert results to DataFrame for easier manipulation
results_df = pd.DataFrame(results)

# Print the results
print(results_df)
