from linear_regression_all_reduce_and_send_recv import *
from linear_regression_compute import *
from linear_regression_attention import *
import torch

class calculate_execution_time(torch.nn.Module):
    def __init__(self, coef_prefilled, intercept_prefilled, coefs, coef_all_reduce, coef_send_recv):
        super(CustomSumFunction, self).__init__()
        
        # Convert parameters to torch tensors if they aren't already
        self.coef_prefilled = torch.tensor(coef_prefilled, dtype=torch.float32, requires_grad=True)
        self.intercept_prefilled = torch.tensor(intercept_prefilled, dtype=torch.float32, requires_grad=True)
        
        # Assuming coefs is a dictionary of model coefficients
        self.coefs = {name: torch.tensor(coef, dtype=torch.float32, requires_grad=True) for name, coef in coefs.items()}
        
        self.coef_all_reduce = torch.tensor(coef_all_reduce, dtype=torch.float32, requires_grad=True)
        self.coef_send_recv = torch.tensor(coef_send_recv, dtype=torch.float32, requires_grad=True)
        