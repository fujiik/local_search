import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time
import random
import os
import networkx as nx

class ising():
	def __init__(self):
		return

	def init_configuration_model(self, n, b, theta_value = 0.5):
		self.n = n
		while True:
			self.theta = np.zeros((self.n, self.n))
			stubs = np.repeat(np.arange(n), b)
			np.random.shuffle(stubs)
			success = True
			for i in range(int(n * b / 2)):
				if stubs[2*i] == stubs[2*i+1] or self.theta[stubs[2*i], stubs[2*i+1]] != 0:
					success = False
					break
				if np.random.random() < 0.5:
					self.theta[stubs[2*i], stubs[2*i+1]] = theta_value
					self.theta[stubs[2*i+1], stubs[2*i]] = theta_value
				else:
					self.theta[stubs[2*i], stubs[2*i+1]] = -theta_value
					self.theta[stubs[2*i+1], stubs[2*i]] = -theta_value
			if success:
				break

	def sample(self, N, n_iter = 100, method = "gibbs"):
		data = np.zeros((N, self.n))
		if method == "gibbs":
			for i in range(N):
				vector_prev = np.zeros(self.n)
				vector_temp = np.zeros(self.n)
				for j in range(n_iter):
					for k in range(self.n):
						vector_concat = np.concatenate((vector_prev[:k], np.array([0]), vector_temp[k+1:]), axis=None)
						cond_prob = 1 / (1 + np.exp(-2 * self.theta[k,:].dot(vector_concat)))
						if np.random.random() < cond_prob:
							vector_temp[k] = 1
						else:
							vector_temp[k] = -1
					vector_prev = vector_temp
				data[i,:] = vector_temp + np.random.normal(size=self.n) * 0.1
			return data
		else:
			print("sampling method name error")
			return

class parameter_estimation():
	def __init__(self):
		return
	
	def set_data(self, data):
		self.data = data
		self.N = data.shape[0]
		self.n = data.shape[1]
		self.sparsity = np.ones((self.n, self.n))
		np.fill_diagonal(self.sparsity, 0)
	
	def set_sparsity(self, sparsity):
		self.sparsity = sparsity

	def solve(self, max_iter = 1000, method="L-BFGS-B"):
		if method == "L-BFGS-B":
			s = int(np.sum(self.sparsity) / 2)
			res = optimize.minimize(self.negative_pseudo_likelihood, x0=np.zeros(s), method=method)
			self.objective = -1 * res.fun
			return self.unravel_solution(res.x)

	def nodewise_solve_one_param(self, v, theta_v, i):
		s = int(np.sum(self.sparsity) / 2)
		self.v = v
		self.i = i
		self.theta_v = theta_v
		res = optimize.minimize(self.nodewise_negative_pseudo_likelihood_with_theta_v, x0=np.zeros(1), method="L-BFGS-B")
		self.nodewise_objective_one_param = -1 * res.fun
		return res.x

	def nodewise_solve(self, v):
		s = int(np.sum(self.nodewise_sparsity))
		self.v = v
		res = optimize.minimize(self.nodewise_negative_pseudo_likelihood, x0=np.zeros(s), method="L-BFGS-B")
		self.nodewise_objective = -1 * res.fun
		return self.unravel_nodewise_solution(res.x)

	def set_nodewise_sparsity(self, sparsity_v):
		self.nodewise_sparsity = sparsity_v

	def unravel_nodewise_solution(self, raveled_solution):
		sol = np.zeros(self.n)
		index = 0
		for i in range(self.n):
			if self.nodewise_sparsity[i] == 1:
				sol[i] = raveled_solution[index]
				index += 1
		return sol

	def nodewise_negative_pseudo_likelihood(self, theta_v_raveled):
		theta_v = self.unravel_nodewise_solution(theta_v_raveled)
		prob = 1 + np.exp(-2 * theta_v.dot(self.data.T) * self.data[:,self.v].T)
		return np.sum(np.log(prob)) / self.N

	def nodewise_negative_pseudo_likelihood_unraveled(self, theta_v):
		prob = 1 + np.exp(-2 * theta_v.dot(self.data.T) * self.data[:,self.v].T)
		return np.sum(np.log(prob)) / self.N

	def nodewise_negative_pseudo_likelihood_with_theta_v(self, theta_vi):
		theta_v = np.zeros(self.n)
		theta_v[:] = self.theta_v[:]
		theta_v[self.v] = 0
		theta_v[self.i] = theta_vi
		prob = 1 + np.exp(-2 * theta_v.dot(self.data.T) * self.data[:,self.v].T)
		return np.sum(np.log(prob)) / self.N

	def pseudo_likelihood(self, solution):
		np.fill_diagonal(solution, 0)
		prob = 1 + np.exp(-2 * solution.T.dot(self.data.T) * self.data.T)
		return -1 * np.sum(np.log(prob)) / self.N

	def d_raveled(self):
		return int(self.n * (self.n + 1) / 2)

	def unravel_solution(self, solution_raveled):
		solution = np.zeros((self.n, self.n))
		index = 0
		for i in range(self.n):
			for j in range(i + 1, self.n):
				if self.sparsity[i,j] == 1:
					solution[i,j] = solution_raveled[index]
					solution[j,i] = solution_raveled[index]
					index += 1
		return solution

	def unravel_solution_test(self):
		m = self.d_raveled()
		print(self.unravel_solution(np.arange(m) + 1))

	def negative_pseudo_likelihood(self, solution_raveled):
		return -1 * self.pseudo_likelihood(self.unravel_solution(solution_raveled))

	def gradient_of_pseudo_likelihood(self, solution):
		a = np.exp(-2 * solution.T.dot(self.data.T) * self.data.T)
		b = a / (1 + a)
		einsum1 = np.einsum('ij,ik->jki', self.data, self.data)
		einsum2 = np.einsum('ij,ik->kji', self.data, self.data)
		grad = np.sum(einsum1 * b + einsum2 * b, axis=2) / self.N * 2
		np.fill_diagonal(grad, 0)
		return grad

class structure_estimation():
	def __init__(self):
		return
	
	def set_data(self, data):
		self.data = data
		self.N = data.shape[0]
		self.n = data.shape[1]

	def set_degree_constraint(self, b):
		assert b.shape == (self.n,)
		self.b = b
	
	def greedy(self):
		X = np.zeros((self.n, self.n))
		current_theta = np.zeros((self.n, self.n))
		current_objective = -10e10
		pe = parameter_estimation()
		pe.set_data(self.data)
		for t in range(self.n * self.n):
			max_objective = -10e10
			max_i = -1
			max_j = -1
			for i in range(self.n):
				for j in range(i + 1, self.n):
					if X[i,j] == 1 or np.sum(X[i]) == self.b[i] or np.sum(X[j]) == self.b[j]:
						continue
					X[i,j] = 1
					X[j,i] = 1
					pe.set_sparsity(X)
					theta = pe.solve(method="L-BFGS-B")
					X[i,j] = 0
					X[j,i] = 0
					if pe.objective > max_objective:
						max_objective = pe.objective
						max_theta = theta
						max_i = i
						max_j = j
			if max_i == -1:
				break
			current_objective = max_objective
			current_theta = max_theta
			X[max_i,max_j] = 1
			X[max_j,max_i] = 1
		return X

	def modular_approximation(self):
		X = np.zeros((self.n, self.n))
		pe = parameter_estimation()
		pe.set_data(self.data)
		objective_initial = - self.n * np.log(2)
		current_theta = np.zeros((self.n, self.n))
		current_objective = -objective_initial

		B = np.zeros((self.n, self.n))
		for i in range(self.n):
			for j in range(i + 1, self.n):
				X[i,j] = 1
				X[j,i] = 1
				pe.set_sparsity(X)
				theta = pe.solve(method="L-BFGS-B")
				B[i, j] = pe.objective - objective_initial
				X[i,j] = 0
				X[j,i] = 0

		if np.all(self.b == 1):
			G = nx.Graph()
			G.add_nodes_from(np.arange(self.n))
			for i in range(self.n):
				for j in range(i + 1, self.n):
					G.add_edge(i, j, weight=B[i,j])
			mate = nx.max_weight_matching(G)
			for key in mate:
				X[key, mate[key]] = 1
		else:
			G = nx.Graph()
			for i in range(self.n):
				for a in range(int(self.b[i])):
					G.add_node(("node", i, a))
			for i in range(self.n):
				for j in range(i + 1, self.n):
					G.add_node(("edge0", i, j))
					G.add_node(("edge1", i, j))
			for i in range(self.n):
				for j in range(i + 1, self.n):
					G.add_edge(("edge0", i, j), ("edge1", i, j), weight=B[i,j])
					for a in range(int(self.b[i])):
						for c in range(int(self.b[j])):
							G.add_edge(("node", i, a), ("edge0", i, j), weight=B[i,j])
							G.add_edge(("node", j, c), ("edge1", i, j), weight=B[i,j])
			mate = nx.max_weight_matching(G)
			for key in mate:
				if key[0] == "edge0" or key[0] == "edge1":
					continue
				i = key[1]
				a = key[2]
				assert i == mate[key][1] or i == mate[key][2]
				if i == mate[key][1]:
					j = mate[key][2]
				else:
					j = mate[key][1]
				if mate[key][0] == "edge0":
					if ("edge1", mate[key][1], mate[key][2]) in mate:
						X[i, j] = 1
						X[j, i] = 1
				elif mate[key][0] == "edge1":
					if ("edge0", mate[key][1], mate[key][2]) in mate:
						X[i, j] = 1
						X[j, i] = 1
		assert np.all(self.b - np.sum(X, axis=1) >= 0)
		return X

	def random(self):
		while True:
			X = np.zeros((self.n, self.n))
			stubs = np.zeros(int(np.sum(self.b)), int)
			i = 0
			for j in range(self.n):
				for r in range(int(self.b[j])):
					stubs[i] = j
					i += 1
			np.random.shuffle(stubs)
			success = True
			for i in range(int(len(stubs) / 2)):
				if stubs[2*i] == stubs[2*i+1] or X[stubs[2*i], stubs[2*i+1]] != 0:
					success = False
					break
				X[stubs[2*i], stubs[2*i+1]] = 1
				X[stubs[2*i+1], stubs[2*i]] = 1
			if success:
				break
		return X

	def oblivious_local_search(self, T):
		X = np.zeros((self.n, self.n))
		current_theta = np.zeros((self.n, self.n))
		current_objective = -10e10
		pe = parameter_estimation()
		pe.set_data(self.data)
		for t in range(T):
			print("iteration", t)
			max_objective = current_objective
			max_i = -1
			max_j = -1
			max_remove_i = -1
			max_remove_j = -1
			for i in range(self.n):
				for j in range(i + 1, self.n):
					if X[i,j] == 1:
						continue
					remove_i = -1
					remove_j = -1
					if np.sum(X[i]) == self.b[i] and np.sum(X[j]) == self.b[j]:
						for remove_i in np.nonzero(X[i])[0]:
							for remove_j in np.nonzero(X[j])[0]:
								X[i,j] = 1
								X[j,i] = 1
								X[i,remove_i] = 0
								X[remove_i,i] = 0
								X[j,remove_j] = 0
								X[remove_j,j] = 0
								pe.set_sparsity(X)
								theta = pe.solve(method="L-BFGS-B")
								X[i,j] = 0
								X[j,i] = 0
								X[i,remove_i] = 1
								X[remove_i,i] = 1
								X[j,remove_j] = 1
								X[remove_j,j] = 1
								if pe.objective > max_objective:
									max_objective = pe.objective
									max_i = i
									max_j = j
									max_remove_i = remove_i
									max_remove_j = remove_j
									max_theta = theta
					elif np.sum(X[i]) == self.b[i] and (not np.sum(X[j]) == self.b[j]):
						for remove_i in np.nonzero(X[i])[0]:
							X[i,j] = 1
							X[j,i] = 1
							X[i,remove_i] = 0
							X[remove_i,i] = 0
							pe.set_sparsity(X)
							theta = pe.solve(method="L-BFGS-B")
							X[i,j] = 0
							X[j,i] = 0
							X[i,remove_i] = 1
							X[remove_i,i] = 1
							if pe.objective > max_objective:
								max_objective = pe.objective
								max_i = i
								max_j = j
								max_remove_i = remove_i
								max_remove_j = remove_j
								max_theta = theta
					elif (not np.sum(X[i]) == self.b[i]) and np.sum(X[j]) == self.b[j]:
						for remove_j in np.nonzero(X[j])[0]:
							X[i,j] = 1
							X[j,i] = 1
							X[j,remove_j] = 0
							X[remove_j,j] = 0
							pe.set_sparsity(X)
							theta = pe.solve(method="L-BFGS-B")
							X[i,j] = 0
							X[j,i] = 0
							X[j,remove_j] = 1
							X[remove_j,j] = 1
							if pe.objective > max_objective:
								max_objective = pe.objective
								max_i = i
								max_j = j
								max_remove_i = remove_i
								max_remove_j = remove_j
								max_theta = theta
					else:
						X[i,j] = 1
						X[j,i] = 1
						pe.set_sparsity(X)
						theta = pe.solve(method="L-BFGS-B")
						X[i,j] = 0
						X[j,i] = 0
						if pe.objective > max_objective:
							max_objective = pe.objective
							max_i = i
							max_j = j
							max_remove_i = remove_i
							max_remove_j = remove_j
							max_theta = theta
					X[i,j] = 0
					X[j,i] = 0
			if max_i == -1:
				print("converged")
				break
			current_objective = max_objective
			current_theta = max_theta
			X[max_i,max_j] = 1
			X[max_j,max_i] = 1
			print("add (%d, %d)" % (max_i, max_j))
			if max_remove_i != -1:
				X[max_i,max_remove_i] = 0
				X[max_remove_i,max_i] = 0
				print("remove (%d, %d)" % (max_i, max_remove_i))
			if max_remove_j != -1:
				X[max_j,max_remove_j] = 0
				X[max_remove_j,max_j] = 0
				print("remove (%d, %d)" % (max_j, max_remove_j))
		return X

	def semi_oblivious_local_search(self, T):
		X = np.zeros((self.n, self.n))
		current_theta = np.zeros((self.n, self.n))
		current_objective = -10e10
		pe = parameter_estimation()
		pe.set_data(self.data)
		for t in range(T):
			print("iteration", t)
			max_objective = current_objective
			max_i = -1
			max_j = -1
			max_remove_i = -1
			max_remove_j = -1
			for i in range(self.n):
				for j in range(i + 1, self.n):
					if X[i,j] == 1:
						continue
					remove_i = -1
					remove_j = -1
					if np.sum(X[i]) == self.b[i]:
						remove_i = np.argmin(np.abs(current_theta[i,:] + (X[i,:] - 1) * 10000))
						X[i,remove_i] = 0
						X[remove_i,i] = 0
					if np.sum(X[j]) == self.b[j]:
						remove_j = np.argmin(np.abs(current_theta[j,:] + (X[j,:] - 1) * 10000))
						X[j,remove_j] = 0
						X[remove_j,j] = 0
					X[i,j] = 1
					X[j,i] = 1
					pe.set_sparsity(X)
					theta = pe.solve(method="L-BFGS-B")
					if remove_i != -1:
						X[i,remove_i] = 1
						X[remove_i,i] = 1
					if remove_j != -1:
						X[j,remove_j] = 1
						X[remove_j,j] = 1
					X[i,j] = 0
					X[j,i] = 0
					if pe.objective > max_objective:
						max_objective = pe.objective
						max_i = i
						max_j = j
						max_remove_i = remove_i
						max_remove_j = remove_j
						max_theta = theta
			if max_i == -1:
				print("converged")
				break
			current_objective = max_objective
			current_theta = max_theta
			X[max_i,max_j] = 1
			X[max_j,max_i] = 1
			print("add (%d, %d)" % (max_i, max_j))
			if max_remove_i != -1:
				X[max_i,max_remove_i] = 0
				X[max_remove_i,max_i] = 0
				print("remove (%d, %d)" % (max_i, max_remove_i))
			if max_remove_j != -1:
				X[max_j,max_remove_j] = 0
				X[max_remove_j,max_j] = 0
				print("remove (%d, %d)" % (max_j, max_remove_j))
		return X

	def compute_M_s3(self):
		N = self.data.shape[0]
		self.M_s3 = 4 * np.sum(np.sqrt(np.sum(self.data ** 2, axis=0)) ** 3) / N

	def non_oblivious_local_search(self, T):
		self.compute_M_s3()
		X = np.zeros((self.n, self.n))
		current_theta = np.zeros((self.n, self.n))
		current_objective = -10e10
		pe = parameter_estimation()
		pe.set_data(self.data)
		for t in range(T):
			print("iteration", t)
			max_gain = 0
			max_i = -1
			max_j = -1
			max_remove_i = -1
			max_remove_j = -1
			for i in range(self.n):
				for j in range(i + 1, self.n):
					if X[i,j] == 1:
						continue
					gain = pe.gradient_of_pseudo_likelihood(current_theta)[i,j] ** 2 / 2 / self.M_s3
					remove_i = -1
					remove_j = -1
					if np.sum(X[i]) == self.b[i]:
						remove_i = np.argmin(np.abs(current_theta[i,:] + (X[i,:] - 1) * 10000))
						gain -= self.M_s3 / 2 * current_theta[i,remove_i] ** 2
					if np.sum(X[j]) == self.b[j]:
						remove_j = np.argmin(np.abs(current_theta[j,:] + (X[j,:] - 1) * 10000))
						gain -= self.M_s3 / 2 * current_theta[j,remove_j] ** 2
					if gain > max_gain:
						max_gain = gain
						max_i = i
						max_j = j
						max_remove_i = remove_i
						max_remove_j = remove_j
			if max_i == -1:
				print("converged")
				break
			X[max_i,max_j] = 1
			X[max_j,max_i] = 1
			print("add (%d, %d)" % (max_i, max_j))
			if max_remove_i != -1:
				X[max_i,max_remove_i] = 0
				X[max_remove_i,max_i] = 0
				print("remove (%d, %d)" % (max_i, max_remove_i))
			if max_remove_j != -1:
				X[max_j,max_remove_j] = 0
				X[max_remove_j,max_j] = 0
				print("remove (%d, %d)" % (max_j, max_remove_j))
			pe.set_sparsity(X)
			current_theta = pe.solve(method="L-BFGS-B")
		return X

def experiment(graph_type, n, arity_list, methods):
	dir_name = './result/{time}_{graph_type}_n{n}'.format(time = time.strftime("%Y%m%d%H%M%S"), graph_type = graph_type, n = n)
	to_make_directory = True

	trial = 10
	N = 100
	T = 200

	colors = {"non_oblivious_local_search": "red", "modular_approximation": "blue", "oblivious_local_search": "pink", "semi_oblivious_local_search": "yellow", "random_residual_greedy": "green", "greedy": "gray", "random": "gray"}
	markers = {"non_oblivious_local_search": "v", "modular_approximation": "h", "oblivious_local_search": "^", "semi_oblivious_local_search": "o", "random_residual_greedy": "s", "greedy": ".", "random": "."}
	lines = {"non_oblivious_local_search": "-", "modular_approximation": "--", "oblivious_local_search": "-", "semi_oblivious_local_search": "-", "random_residual_greedy": "--", "greedy": ".-", "random": ".-"}

	result = np.zeros((trial, len(arity_list), len(methods)))
	running_time = np.zeros((trial, len(arity_list), len(methods)))
	for tr in range(trial):
		instance = ising()
		random.seed(20190926+tr)
		np.random.seed(20190926+tr)
		if graph_type == "line":
			instance.init_line_graph(n, theta_value = 0.5)
		elif graph_type == "grid":
			instance.init_grid_graph(n, theta_value = 0.5)
		elif graph_type == "configuration":
			instance.init_configuration_model(n, np.max(arity_list), theta_value = 0.5)
		data = instance.sample(N)

		pe = parameter_estimation()
		pe.set_data(data)
		objective_initial = - n * np.log(2)
		print(objective_initial)

		X = (instance.theta != 0).astype(int)
		pe.set_sparsity(X)
		theta = pe.solve(method="L-BFGS-B")
		objective_optimal = pe.objective
		print(objective_optimal)

		for arity_index in range(len(arity_list)):
			arity = arity_list[arity_index]

			se = structure_estimation()
			se.set_data(data)
			se.set_degree_constraint(np.ones(n) * arity)

			for m in range(len(methods)):
				method = methods[m]
				print("start %s" % method)
				start_time = time.time()
				if method == "semi_oblivious_local_search":
					X = se.semi_oblivious_local_search(T)
				elif method == "oblivious_local_search":
					X = se.oblivious_local_search(T)
				elif method == "non_oblivious_local_search":
					X = se.non_oblivious_local_search(T)
				elif method == "greedy":
					X = se.greedy()
				elif method == "modular_approximation":
					X = se.modular_approximation()
				elif method == "random":
					X = se.random()
				running_time[tr, arity_index, m] = time.time() - start_time
				pe.set_sparsity(X)
				pe.solve()
				print(pe.objective)
				result[tr, arity_index, m] = (pe.objective - objective_initial) / (objective_optimal - objective_initial)
				print("trial %02d, arity %d, method %s: objective %f" % (tr, arity, method, result[tr, arity_index, m]))

	if to_make_directory:
		to_make_directory = False
		os.mkdir(dir_name)

	axis = (arity_list * n / 2).astype(int)
	for m in range(len(methods)):
		method = methods[m]
		filename = dir_name + "/result_{method}_{graph_type}_n{n}.txt".format(method = method, graph_type = graph_type, n = n)
		np.savetxt(filename, result[:,:,m], delimiter=",", fmt="%1.5f")
		plt.plot(axis, np.mean(result[:,:,m], axis=0), label=method, color=colors[method], marker=markers[method])
	plt.legend()
	plt.savefig(dir_name + "/plot_{graph_type}_n{n}.pdf".format(graph_type = graph_type, n = n))
	plt.show()

	for m in range(len(methods)):
		method = methods[m]
		filename = dir_name + "/time_{method}_{graph_type}_n{n}.txt".format(method = method, graph_type = graph_type, n = n)
		np.savetxt(filename, running_time[:,:,m], delimiter=",", fmt="%1.5f")
		plt.plot(axis, np.mean(running_time[:,:,m], axis=0), label=method, color=colors[method], marker=markers[method])
	plt.yscale("log")
	plt.legend()
	plt.savefig(dir_name + "/time_{graph_type}_n{n}.pdf".format(graph_type = graph_type, n = n))
	plt.show()

if __name__ == '__main__':
	print("experiment with |V| = 10")
	arity_list = 1 + np.arange(5)
	methods = ["oblivious_local_search", "semi_oblivious_local_search", "non_oblivious_local_search", "modular_approximation", "random"]
	experiment("configuration", 10, arity_list, methods)

	print("experiment with |V| = 20")
	arity_list = 1 + np.arange(7)
	methods = ["non_oblivious_local_search", "modular_approximation", "random"]
	experiment("configuration", 20, arity_list, methods)
