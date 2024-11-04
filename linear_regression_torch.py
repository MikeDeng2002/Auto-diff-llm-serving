import pandas as pd
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_and_evaluate(data, feature_columns, target_column, num_epochs=100, learning_rate=0.01):
    X = data[feature_columns].values
    y = data[target_column].values.reshape(-1, 1)
    
    # Check for any NaNs in X or y after filtering
    if pd.isna(X).any() or pd.isna(y).any():
        raise ValueError(f"NaN values detected in features or target column {target_column}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Initialize model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = mean_squared_error(y_test.numpy(), y_pred.numpy())
    
    # Extract model parameters
    coefficients = model.linear.weight.detach().numpy()
    intercept = model.linear.bias.detach().numpy()

    return model, mse, coefficients, intercept

if __name__ == "__main__":
    compute_file = '/Users/dengtianze/Documents/GitHub/vidur/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv'
    compute = pd.read_csv(compute_file)

    # Define model names and target columns
    model_names = [
        "attn_pre_proj",
        "attn_post_proj",
        "mlp_up_proj",
        "mlp_down_proj",
        "mlp_act",
        "input_layernorm",
        "post_attention_layernorm",
        "attn_rope",
        "add",
    ]
    target_columns = [f"time_stats.{model_name}.median" for model_name in model_names]

    # Drop rows with NaNs in the target columns
    compute = compute.dropna(subset=target_columns)
    print("Data shape after filtering NaNs in target columns:", compute.shape)

    # Convert very small values to zero (optional, for values near zero)
    compute[target_columns] = compute[target_columns].applymap(
        lambda x: 0 if abs(x) < 1e-15 else x
    )

    # Check for NaNs in specific columns to ensure clean data
    for col in target_columns:
        if col in compute.columns:
            nan_count = compute[col].isna().sum()
            if nan_count > 0:
                print(f"Column '{col}' still contains {nan_count} NaN values.")
            else:
                print(f"Column '{col}' is free of NaNs.")

    feature_columns = ['num_tokens']
    models = {}
    coefs = {}
    intercepts = {}

    for model_name in model_names:
        target_col = f"time_stats.{model_name}.median"
        if target_col not in compute.columns:
            print(f"Warning: Target column {target_col} not found in data. Skipping {model_name}.")
            continue

        try:
            model, mse, coef, intercept = train_and_evaluate(compute, feature_columns, target_col)
            models[model_name] = model
            coefs[model_name] = coef
            intercepts[model_name] = intercept

            print(f"Mean Squared Error for {model_name}: {mse}")
            print(f"Coefficients for {model_name}: {coef}")
            print(f"Intercept for {model_name}: {intercept}")

        except ValueError as e:
            print(f"Error for {model_name}: {e}")
