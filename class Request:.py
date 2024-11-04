class Request:
	n_tokens: int
	is_prefill: torch.tensor(int)
	duration_alive = torch.tensor(int)

class Server:
	queue_length: tensor(int)
	prefill_jobs = []
	decode_jobs = []
	remaining_service_time = torch.tensor(1e-6)


model = Model()

next_arrival_time = torch.tensor(float)


def request_arrival(request):

	features = torch.concat([server.queue_length for server in servers] + [other features ...] + [request features])

	routing_decision = model(features)

	chosen_server = torch.argmax(routing_decision)

	server = servers[chosen_server]

	server.prefill_jobs.append(reqeust)

	for server_idx in len(servers):
		servers[server_idx].queue_length += routing_decision[server_idx]




def batch_and_serve(server_idx):

	is_prefill_tensor = torch.concat([job[job_idx].is_prefill for job_idx in len(prefill_jobs)] + [... in decode_jobs])

	n_tokens_tensor = ...

	n_token_tensor_rolling_sum = torch.cumsum(n_tokens_tensor) #to calculate num_blocks allocated if up to i-th prefill job served
	n_blocks_used = n_token_tensor_rolling_sum / block_size
	can_be_served = n_blocks_used <= max_block_num
	can_be_served = can_be_served * is_prefill_tensor

	duration_alive_tensor = torch.concat([job[job_idx].duration_alive for job_idx in len(prefill_jobs)]) + [... in decode_jobs]
	duration_alive_tensor = (1 - is_prefill_tensor) * duration_alive_tensor
	chosen_decode_jobs = top_k_softmax(duration_alive_tensor, max_batch_size)


	is_prefill_batch = can_be_served.sum() / can_be_served.sum().detach()

	job_priority = can_be_served +  (1 - is_prefill_batch) * chosen_decode_jobs



	execution_time = calculate_execution_time(...)

	servers[server_idx].remaining_service_time += execution_time


	is_prefill_tensor -= can_be_served

	#remove prefill jobs from prefill_jobs and append to decode_jobs here

	n_tokens_tensor -= (1 - is_prefill_batch) * chosen_decode_jobs



while True:


	next_event_idx = argmin(torch.concat([next_arrival_time] + [server.remaining_service_time for server in servers]))

	if next_event_idx == 0:
		request_arrival()
		#and update all time related variables
	else:
		batch_and_serve(severs[next_time_idx - 1])

def calculate_executation_time(is_prefilled_batch,batch_size,kv_cahse_size):
	if is_prefilled_batch:
		prefilled_chunk_size = batch_size ** 2
	time = kv
	




