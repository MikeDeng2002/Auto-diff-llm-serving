import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from linear_regression import train_and_evaluate

compute_file = '/Users/dengtianze/Documents/GitHub/vidur/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv'
compute = pd.read_csv(compute_file)
compute = compute[compute['num_tensor_parallel_workers'] == 1]
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
feature_columns = ['num_tokens']
models = {}

coefs = {}
intercepts = {}
for model_name in model_names:
    target_col=f"time_stats.{model_name}.median"
    models[model_name], mse, coefs[model_name], intercepts[model_name] = train_and_evaluate(compute, feature_columns, target_col)
    
    
coef_compute = sum(coefs.values())
intercept_compute = sum(intercepts.values())
print(coef_compute)
print(intercept_compute)

