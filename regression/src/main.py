import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time
import random
import heapq
import os

class linear():
	def __init__(self):
		return

	def init_partition_matroid(self, n, m, arity):
		assert int(n / m) * m == n
		self.n = n
		self.m = m # number of partitions
		self.sp = int(n / m) # size of partition
		self.arity = arity
		self.support = np.zeros(self.n)
		self.support_list = []
		for p in range(m):
			indices = p * self.sp + np.random.choice(self.sp, self.arity, replace=False)
			for index in indices:
				self.support[index] = 1
				self.support_list.append(index)
		self.k = int(np.sum(self.support))
		self.theta = np.zeros(self.n)
		self.theta[self.support_list] = np.random.randn(self.k)
	
	def sample(self, N):
		self.N = N
		self.A = np.random.uniform(size = (self.n, self.N))
		for i in range(self.n):
			self.A[i,:] -= np.mean(self.A[i,:])
			self.A[i,:] /= np.linalg.norm(self.A[i,:])
		self.y = self.A.T.dot(self.theta) + np.random.normal(size=self.N) * 0.2
		self.y /= np.linalg.norm(self.y)
		return (self.A, self.y)

class parameter_estimation():
	def __init__(self):
		return
	
	def set_data(self, A, y):
		self.A = A
		self.y = y
		self.n = A.shape[0]
		self.N = A.shape[1]
		self.sparsity = np.array(np.arange(self.n))
	
	def set_sparsity(self, sparsity):
		self.sparsity = sparsity

	def solve(self):
		solution = np.linalg.pinv(self.A[self.sparsity,:].dot(self.A[self.sparsity,:].T)).dot(self.A[self.sparsity,:].dot(self.y))
		theta = np.zeros(self.n)
		theta[self.sparsity] = solution
		y_hat = theta.dot(self.A)
		self.objective = self.y.dot(self.y) - (self.y - y_hat).dot(self.y - y_hat)
		return theta

class structure_estimation():
	def __init__(self):
		return
	
	def set_data(self, A, y):
		self.A = A
		self.y = y
		self.n = A.shape[0]
		self.N = A.shape[1]

	def set_partition_matroid_constraint(self, m, arity):
		self.m = m
		self.arity = arity
		self.sp = int(self.n / self.m)

	def compute_M_s2(self):
		max_cor = 1
		for i in range(self.n):
			for j in range(self.n):
				cor = self.A[i,:].dot(self.A[i,:])
				if cor > max_cor:
					max_cor = cor
		self.M_s2 = 1 + max_cor
	
	def non_oblivious_local_search(self, T):
		self.compute_M_s2()
		support = np.zeros(self.n)
		partition_flag = np.full((self.m, self.arity), -1)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		for t in range(T):
			print("iteration", t)
			max_gain = 0
			max_i = -1
			max_remove_i = -1
			max_remove_index = -1
			for i in range(self.n):
				if support[i] == 1:
					continue
				gain = self.A[i,:].dot(self.y - self.A.T.dot(current_theta)) ** 2 / 2 / self.M_s2
				remove_i = -1
				remove_index = -1
				if np.all(partition_flag[i_to_p[i],:] != -1):
					remove_index = np.argmin(current_theta[partition_flag[i_to_p[i],:]] ** 2)
					remove_i = partition_flag[i_to_p[i], remove_index]
					gain -= self.M_s2 / 2 * current_theta[remove_i] ** 2
				if gain > max_gain:
					max_gain = gain
					max_i = i
					max_remove_i = remove_i
					max_remove_index = remove_index
			if max_i == -1:
				print("converged")
				break
			support[max_i] = 1
			for arity_index in range(self.arity):
				if partition_flag[i_to_p[max_i], arity_index] == -1:
					partition_flag[i_to_p[max_i], arity_index] = max_i
					break
			print("add %d" % max_i)
			if max_remove_i != -1:
				support[max_remove_i] = 0
				partition_flag[i_to_p[max_i], max_remove_index] = max_i
				print("remove %d" % max_remove_i)
			support_list = []
			for i in range(self.n):
				if support[i] == 1:
					support_list.append(i)
			pe.set_sparsity(support_list)
			current_theta = pe.solve()
		return support

	def semi_oblivious_local_search(self, T):
		support = np.zeros(self.n)
		partition_flag = np.full((self.m, self.arity), -1)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		for t in range(T):
			print("iteration", t)
			max_objective = current_objective
			max_theta = np.zeros(self.n)
			max_i = -1
			max_remove_i = -1
			max_remove_index = -1
			for i in range(self.n):
				if support[i] == 1:
					continue
				remove_i = -1
				remove_index = -1
				if np.all(partition_flag[i_to_p[i],:] != -1):
					remove_index = np.argmin(current_theta[partition_flag[i_to_p[i],:]] ** 2)
					remove_i = partition_flag[i_to_p[i], remove_index]
					support[remove_i] = 0
				support[i] = 1
				support_list = []
				for j in range(self.n):
					if support[j] == 1:
						support_list.append(j)
				pe.set_sparsity(support_list)
				theta = pe.solve()
				objective = pe.objective
				support[i] = 0
				if remove_i != -1:
					support[remove_i] = 1
				if objective > max_objective:
					max_objective = objective
					max_theta = theta
					max_i = i
					max_remove_i = remove_i
					max_remove_index = remove_index
			if max_i == -1:
				print("converged")
				break
			support[max_i] = 1
			for arity_index in range(self.arity):
				if partition_flag[i_to_p[max_i], arity_index] == -1:
					partition_flag[i_to_p[max_i], arity_index] = max_i
					break
			print("add %d" % max_i)
			if max_remove_i != -1:
				support[max_remove_i] = 0
				partition_flag[i_to_p[max_i], max_remove_index] = max_i
				print("remove %d" % max_remove_i)
			current_objective = max_objective
			current_theta = max_theta
		return support

	def oblivious_local_search(self, T):
		support = np.zeros(self.n)
		partition_flag = np.full((self.m, self.arity), -1)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		for t in range(T):
			print("iteration", t)
			max_objective = current_objective
			max_theta = np.zeros(self.n)
			max_i = -1
			max_remove_i = -1
			max_remove_index = -1
			for i in range(self.n):
				if support[i] == 1:
					continue
				remove_i = -1
				remove_index = -1
				if np.all(partition_flag[i_to_p[i],:] != -1):
					for remove_index in range(self.arity):
						remove_i = partition_flag[i_to_p[i], remove_index]
						support[i] = 1
						support[remove_i] = 0
						support_list = []
						for j in range(self.n):
							if support[j] == 1:
								support_list.append(j)
						pe.set_sparsity(support_list)
						theta = pe.solve()
						objective = pe.objective
						support[i] = 0
						support[remove_i] = 1
						if objective > max_objective:
							max_objective = objective
							max_theta = theta
							max_i = i
							max_remove_i = remove_i
							max_remove_index = remove_index
				else:
					support[i] = 1
					support_list = []
					for j in range(self.n):
						if support[j] == 1:
							support_list.append(j)
					pe.set_sparsity(support_list)
					theta = pe.solve()
					objective = pe.objective
					support[i] = 0
					if objective > max_objective:
						max_objective = objective
						max_theta = theta
						max_i = i
						max_remove_i = remove_i
						max_remove_index = remove_index
			if max_i == -1:
				print("converged")
				break
			support[max_i] = 1
			for arity_index in range(self.arity):
				if partition_flag[i_to_p[max_i], arity_index] == -1:
					partition_flag[i_to_p[max_i], arity_index] = max_i
					break
			print("add %d" % max_i)
			if max_remove_i != -1:
				support[max_remove_i] = 0
				partition_flag[i_to_p[max_i], max_remove_index] = max_i
				print("remove %d" % max_remove_i)
			current_objective = max_objective
			current_theta = max_theta
		return support

	def greedy(self):
		support = np.zeros(self.n)
		partition_flag = np.full(self.m, -1)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		for t in range(self.m):
			print("iteration", t)
			max_objective = current_objective
			max_theta = np.zeros(self.n)
			max_i = -1
			for i in range(self.n):
				if support[i] == 1:
					continue
				support[i] = 1
				support_list = []
				for j in range(self.n):
					if support[j] == 1:
						support_list.append(j)
				pe.set_sparsity(support_list)
				theta = pe.solve()
				objective = pe.objective
				support[i] = 0
				if objective > max_objective:
					max_objective = objective
					max_theta = theta
					max_i = i
			if max_i == -1:
				print("converged")
				break
			support[max_i] = 1
			partition_flag[i_to_p[max_i]] = max_i
			print("add %d" % max_i)
			current_objective = max_objective
			current_theta = max_theta
		return support

	def random_residual_greedy(self):
		support = np.zeros(self.n)
		partition_counter = np.full(self.m, 0)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		for t in range(self.m * self.arity):
			print("iteration", t)
			r = np.random.randint(self.m * self.arity - t)
			selected_p = -1
			for p in range(self.m):
				if self.arity - partition_counter[p] <= r:
					r -= self.arity - partition_counter[p]
					continue
				else:
					selected_p = p
					break

			max_objective = current_objective
			max_theta = np.zeros(self.n)
			max_i = -1
			gain_queue = []
			for i in range(self.n):
				if i_to_p[i] != selected_p or support[i] == 1:
					continue
				support[i] = 1
				support_list = []
				for j in range(self.n):
					if support[j] == 1:
						support_list.append(j)
				pe.set_sparsity(support_list)
				theta = pe.solve()
				objective = pe.objective
				support[i] = 0
				heapq.heappush(gain_queue, (-1 * objective, i))

			for rank in range(r+1):
				(minus_max_objective, max_i) = heapq.heappop(gain_queue)
			max_objective = -1 * minus_max_objective
			print(max_objective)

			if max_i == -1:
				print("converged")
				break
			support[max_i] = 1
			partition_counter[i_to_p[max_i]] += 1
			print("add %d" % max_i)
			current_objective = max_objective
		return support

	def modular_approximation(self):
		partition_counter = np.full(self.m, 0)
		i_to_p = np.zeros(self.n, int)
		for i in range(self.m):
			for j in range(self.sp):
				i_to_p[i*self.sp+j] = i
		current_theta = np.zeros(self.n)
		current_objective = 0
		pe = parameter_estimation()
		pe.set_data(self.A, self.y)
		b = np.abs(self.A.dot(self.y))
		ranking = np.argsort(-1 * b)
		support = np.zeros(self.n)
		for j in range(self.n):
			i = ranking[j]
			if partition_counter[i_to_p[i]] == self.arity:
				continue
			print("add", i)
			support[i] = 1
			partition_counter[i_to_p[i]] += 1
		return support

def experiment(matroid_type, n, N, num_c, num_p, methods):
	dir_name = './result/{time}_{matroid_type}_n{n}'.format(time = time.strftime("%Y%m%d%H%M%S"), matroid_type = matroid_type, n = n)
	to_make_directory = True

	trial = 10
	arity_list = 1 + np.arange(num_p)
	T = 200

	colors = {"non_oblivious_local_search": "red", "modular_approximation": "blue", "oblivious_local_search": "pink", "semi_oblivious_local_search": "yellow", "random_residual_greedy": "green", "greedy": "gray"}
	markers = {"non_oblivious_local_search": "v", "modular_approximation": "h", "oblivious_local_search": "^", "semi_oblivious_local_search": "o", "random_residual_greedy": "s", "greedy": "."}
	lines = {"non_oblivious_local_search": "-", "modular_approximation": "--", "oblivious_local_search": "-", "semi_oblivious_local_search": "-", "random_residual_greedy": "--", "greedy": ".-"}

	result = np.zeros((trial, len(arity_list), len(methods)))
	running_time = np.zeros((trial, len(arity_list), len(methods)))
	random.seed(20190926)
	np.random.seed(20190926)
	for tr in range(trial):
		instance = linear()
		instance.init_partition_matroid(n, num_c, num_p)
		A, y = instance.sample(N)

		for arity_index in range(len(arity_list)):
			arity = arity_list[arity_index]
			se = structure_estimation()
			se.set_data(A, y)
			se.set_partition_matroid_constraint(num_c, arity)
			pe = parameter_estimation()
			pe.set_data(A, y)

			for m in range(len(methods)):
				method = methods[m]
				print("start %s" % method)
				start_time = time.time()
				if method == "non_oblivious_local_search":
					support = se.non_oblivious_local_search(T)
				elif method == "semi_oblivious_local_search":
					support = se.semi_oblivious_local_search(T)
				elif method == "oblivious_local_search":
					support = se.oblivious_local_search(T)
				elif method == "random_residual_greedy":
					support = se.random_residual_greedy()
				elif method == "greedy":
					support = se.greedy()
				elif method == "modular_approximation":
					support = se.modular_approximation()
				running_time[tr, arity_index, m] = time.time() - start_time
				support_list = []
				for j in range(n):
					if support[j] == 1:
						support_list.append(j)
				pe.set_sparsity(support_list)
				pe.solve()
				result[tr, arity_index, m] = pe.objective
				print("trial %02d, s %d, method %s: residual %f" % (tr, arity * num_c, method, result[tr, arity_index, m]))

	if to_make_directory:
		to_make_directory = False
		os.mkdir(dir_name)

	axis = arity_list * num_c
	for m in range(len(methods)):
		method = methods[m]
		filename = dir_name + "/result_{method}_{matroid_type}_n{n}.txt".format(method = method, matroid_type = matroid_type, n = n)
		np.savetxt(filename, result[:,:,m], delimiter=",", fmt="%1.5f")
		plt.plot(axis, np.mean(result[:,:,m], axis=0), label=method, color=colors[method], marker=markers[method])
	plt.legend()
	plt.savefig(dir_name + "/result_{matroid_type}_n{n}.pdf".format(matroid_type = matroid_type, n = n))
	plt.show()


	for m in range(len(methods)):
		method = methods[m]
		filename = dir_name + "/time_{method}_{matroid_type}_n{n}.txt".format(method = method, matroid_type = matroid_type, n = n)
		np.savetxt(filename, running_time[:,:,m], delimiter=",", fmt="%1.5f")
		plt.plot(axis, np.mean(running_time[:,:,m], axis=0), label=method, color=colors[method], marker=markers[method])
	plt.yscale("log")
	plt.legend()
	plt.savefig(dir_name + "/time_{matroid_type}_n{n}.pdf".format(matroid_type = matroid_type, n = n))
	plt.show()

if __name__ == '__main__':
	print("experiment with n = 200")
	methods = ["oblivious_local_search", "semi_oblivious_local_search", "non_oblivious_local_search", "random_residual_greedy", "modular_approximation"]
	experiment("partition", 200, 50, 5, 5, methods)
	
	print("experiment with n = 1000")
	methods = ["non_oblivious_local_search", "random_residual_greedy", "modular_approximation"]
	experiment("partition", 1000, 100, 10, 10, methods)
