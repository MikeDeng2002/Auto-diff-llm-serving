from linear_regression_all_reduce_and_send_recv import *
from linear_regression_compute import *
from linear_regression_attention import *
import torch

class calculate_execution_time(torch.nn.Module):
    def __init__(self, coef_prefilled, intercept_prefilled, coef_decoded, intercept_decoded, coef_compute, intercept_compute, coef_all_reduce, intercept_all_reduce, coef_send_recv, intercept_send_recv, batch_size, kv_cache_size):
        super(CustomSumFunction, self).__init__()
        
        # Convert parameters to torch tensors if they aren't already
        self.coef_prefilled = torch.tensor(coef_prefilled, dtype=torch.float32, requires_grad=True)
        self.intercept_prefilled = torch.tensor(intercept_prefilled, dtype=torch.float32, requires_grad=True)
        self.coef_decoded = torch.tensor(coef_decoded, dtype=torch.float32, requires_grad=True) 
        self.intercept_decoded = torch.tensor(intercept_decoded, dtype=torch.float32, requires_grad=True)
        self.coef_compute = torch.tensor(coef_compute, dtype=torch.float32, requires_grad=True)
        self.intercept_compute = torch.tensor(intercept_compute, dtype=torch.float32, requires_grad=True)
        self.coef_all_reduce = torch.tensor(coef_all_reduce, dtype=torch.float32, requires_grad=True)   
        self.intercept_all_reduce = torch.tensor(intercept_all_reduce, dtype=torch.float32, requires_grad=True) 
        self.coef_send_recv = torch.tensor(coef_send_recv, dtype=torch.float32, requires_grad=True)
        self.intercept_send_recv = torch.tensor(intercept_send_recv, dtype=torch.float32, requires_grad=True)
        self.batch_size = torch.tensor(batch_size, dtype=torch.float32, requires_grad=True)
        self.kv_cache_size = torch.tensor(kv_cache_size, dtype=torch.float32, requires_grad=True)
    def prefilled_time(self):
        prefilled_chunk_size_quare = batch_size ** 2
        time = self.kv_cache_size * self.coef_prefilled[0]+prefilled_chunk_size_quare * self.coef_prefilled[1] + self.intercept_prefilled + self.batch_size *(self.coef_compute+ self.coef_send_recv+self.coef_all_reduce)+self.intercept_compute+self.intercept_send_recv+self.intercept_all_reduce
        return time
    def decoded_time(self):
        time = self.kv_cache_size * self.coef_decoded[0]+self.batch_size * self.coef_decoded[1] + self.intercept_decoded + self.batch_size *(self.coef_compute+ self.coef_send_recv+self.coef_all_reduce)+self.intercept_compute+self.intercept_send_recv+self.intercept_all_reduce
        return time