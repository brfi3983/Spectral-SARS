import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import networkx as nx
import pandas as pd
from SIR_epidemic import *

plt.style.use('ggplot')

# Importing Networks
pres_filenames = ['A_pres_InVS13', 'A_pres_InVS15', 'A_pres_LH10', 'A_pres_LyonSchool', 'A_pres_SFHH', 'A_pres_Thiers13']
cont_filenames = ['A_lnVS13', 'A_lnVS15', 'A_LH10', 'A_LyonSchool', 'A_SFHH', 'A_Thiers13']

# Choosing Contact vs. Presence Network
contact = False

# Defining variables for type of network
if contact == True:
	filenames = cont_filenames
	folder = 'contact_data'
	title = 'Contact Networks'
	color = 'red'
	color2 = 'orange'
else:
	filenames = pres_filenames
	folder = 'presence_data'
	title = 'Presence Networks'
	color = 'teal'
	color2 = 'steelblue'

# ========================================================
class NetworkStat():
	def __init__(self, A):
		self.data = A
		self.n = A.shape[0]
		self.m = int(np.sum(abs(A))/2)

	def averageDegree(self):
		self.d_avg = int((self.m / self.n)*2)

	def density(self):
		return 2 * self.m / (self.n*(self.n - 1))

	def domEig(self):
		w, v = LA.eig(self.data)
		return np.amax([np.amax(w), np.amax(-w)]).real

	def degreeDis(self):
		# self.degD = np.sum(self.data, axis = 1)
		G = nx.Graph(self.data)
		self.degD = [G.degree(n) for n in G.nodes()]

	def cluster(self):

		self.cc_vec = list(nx.clustering(nx.Graph(self.data)).values())
		self.cc = np.mean(self.cc_vec)

# ========================================================
def main():

	# Figures for two plots
	fig1 = plt.figure(figsize=(16,9))
	fig2 = plt.figure(figsize=(16,9))

	# Figure titles
	fig1.suptitle(f'Degree\'s for {title}', fontsize=30)
	fig2.suptitle(f'Clustering for {title}', fontsize=30)

	# Cycle through the networks, find relevant stats, and then add them to the subplot
	density = []
	dom_eig = []
	for i, name in enumerate(filenames):

		# Importing Network
		A = np.genfromtxt('./' + str(folder) + '/' + str(name) + '.csv', delimiter=',')

		# Turning matrix into an Object
		print('\n========================================================')
		print(f' --- {name} --- ')
		net = NetworkStat(A)
		print(f' Number of Vertices: {net.n} | Number of Edges: {net.m}')

		# Getting statistics
		density.append(net.density())
		dom_eig.append(net.domEig())

		# Degree Distribution
		net.degreeDis()
		deg_vector = net.degD


		L = np.diag(deg_vector) - A

		eigvals, eigvect = LA.eig(L)
		eigenvals_sorted_indices = np.argsort(eigvals)
		eigenvals_sorted = eigvals[eigenvals_sorted_indices]

		zero_eigenvals_index = np.argwhere(abs(eigvals) < 1e-3)
		print(eigvals[zero_eigenvals_index])
		plt.plot(range(1,eigvals.size + 1), eigenvals_sorted)

		# Clustering coefficient and distribution (NOTICE: .cluster() must be run after degree distribution)
		net.cluster()
		cc_vec, cc = net.cc_vec, net.cc
		print(f'Cluster Coefficient Average: {cc:0.3f}')
		# Plotting Degree Histograms
		ax1 = fig1.add_subplot(3, 2, i + 1)
		ax1.hist(deg_vector, bins=net.n, ec='white', density=True, color=color)
		ax1.set_title(name)
		ax1.set_xlabel('Degree')

		# Plotting Clustering Histograms
		ax2 = fig2.add_subplot(3, 2, i + 1)
		ax2.hist(cc_vec, bins=net.n, ec='white', color=color2)
		ax2.set_title(name)
		ax2.set_xlabel('Clustering Coefficient')
		model = SIR_class(A)


		B = 4e-4
		T = 1e3

		sims_R = {}
		p0 = {}

		sim_R = np.empty(0)
		mu = 100 * B

		for i in range(10):
			[S, I, R] = model.SIR(B, mu, T, vaccinated=2)
			sim_R = np.append(sim_R, R)

		sims_R[str(k)] = sim_R
		p0[str(k)] = B * net.d_avg / mu

		# Plotting distribution of simulations for each k (for each dataset - CREATE NEW FIGURE or overlay histograms...)
		ax2 = fig2.add_subplot(3, 2, k + 1)
		ax2.set_title(r'$p_0$ value of ' + str(p0[str(k)]))
		ax2.hist(sims_R[str(k)], bins=15, ec='white')

	# Cleaning up figure
	fig1.subplots_adjust(hspace=0.4, wspace = 0.1)
	fig2.subplots_adjust(hspace=0.4, wspace = 0.1)


	d = {'Density': density, 'Dominent Eigenvalue': dom_eig}
	df = pd.DataFrame(data=d)

# ========================================================
if __name__ == "__main__":
	main()