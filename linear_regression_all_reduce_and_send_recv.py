import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from linear_regression import train_and_evaluate

all_reduce_file = '/Users/dengtianze/Documents/GitHub/vidur/data/profiling/network/a100_pairwise_nvlink/all_reduce.csv'
all_reduce_df = pd.read_csv(all_reduce_file)
all_reduce_df = all_reduce_df.dropna()
all_reduce_df = all_reduce_df[all_reduce_df['num_workers'] == 2]
all_reduce_df['num_tokens']=all_reduce_df['size']/2/4096
feature_columns = ['num_tokens']
target_column = ['time_stats.all_reduce.median']
model, mse, coef_all_reduce, intercept_all_reduce = train_and_evaluate(all_reduce_df, feature_columns, target_column)
print(f"Mean Squared Error for : {mse}")
print(f"Coefficients for : {coef_all_reduce}")
print(f"Intercept for : {intercept_all_reduce}")

send_recv_file = '/Users/dengtianze/Documents/GitHub/vidur/data/profiling/network/a100_pairwise_nvlink/send_recv.csv'
send_recv_df = pd.read_csv(send_recv_file)  
send_recv_df = send_recv_df.dropna()
send_recv_df = send_recv_df[send_recv_df['devices_per_node'] == 2]
send_recv_df['num_tokens'] = send_recv_df['size']/2/4096
feature_columns = ['num_tokens']
target_column = ['time_stats.send_recv.median']
model, mse, coef_send_recv, intercept_send_recv = train_and_evaluate(send_recv_df, feature_columns, target_column)
print(f"Mean Squared Error for : {mse}")
print(f"Coefficients for : {coef_send_recv}")
print(f"Intercept for : {intercept_send_recv}")