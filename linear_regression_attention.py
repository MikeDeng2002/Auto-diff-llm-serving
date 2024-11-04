import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from linear_regression import train_and_evaluate

attention_file = '/Users/dengtianze/Documents/GitHub/vidur/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv'
attention = pd.read_csv(attention_file)
attention['prefilled_chunk_size_squared'] = attention['prefill_chunk_size'] ** 2
attention_prefilled = attention[attention['is_prefill'] == True]
attention_not_prefilled = attention[attention['is_prefill'] == False]

feature_columns_decoded_decoded = ['kv_cache_size', 'batch_size']
target_column_prefilled_decoded = 'time_stats.attn_output_reshape.median'  # Should be a string, not a list
model_decode, mse, coef_decode, intercept_decode = train_and_evaluate(attention_not_prefilled, feature_columns_decoded_decoded, target_column_prefilled_decoded)
print(f"Mean Squared Error for decoded: {mse}")
print(f"Coefficients for decoded: {coef_decode}")
print(f"Intercept for decoded: {intercept_decode}")

feature_columns_decoded_prefilled = ['kv_cache_size', 'prefilled_chunk_size_squared']
target_column_prefilled_decoded = 'time_stats.attn_output_reshape.median'  # Should be a string, not a list
model_prefill, mse, coef_prefill, intercept_prefill = train_and_evaluate(attention_prefilled, feature_columns_decoded_prefilled, target_column_prefilled_decoded)
print(f"Mean Squared Error for prefilled: {mse}")
print(f"Coefficients for prefilled: {coef_prefill[0]}")
print(f"Intercept for prefilled: {intercept_prefill}")

