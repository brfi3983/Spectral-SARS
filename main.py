import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import linalg as LA
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
else:
	filenames = pres_filenames
	folder = 'presence_data'
	title = 'Presence Networks'
	color = 'teal'

# ========================================================
class NetworkStat():
	def __init__(self, A):
		self.data = A
		self.n = A.shape[0]
		self.m = int(np.sum(abs(A))/2)

	def averageDegree(self):
		self.d_avg = int((self.m / self.n)*2)

	def density(self):
		self.den = 2 * self.m / (self.n(self.n - 1))

	def domEig(self):
		w, v = LA.eig(self.data)
		self.lam = np.amax([np.amax(w), np.amax(-w)]).real

	def degreeDis(self):
		self.degD = np.sum(self.data, axis = 1)

	def clusterCoeff(self):
		pass

# ========================================================
def main():

	# Initializing figures
	fig1 = plt.figure(figsize=(16,9))
	fig2 = plt.figure(figsize=(16,9))
	fig3 = plt.figure(figsize=(16,9))

	# Titles for figures
	fig1.suptitle(title, fontsize=30)
	fig2.suptitle(f'Eigenvalues for {title}', fontsize=30)
	fig3.suptitle(f'Graph\'s for {title}', fontsize=30)

	# Cycle through the networks, find relevant stats, and then add them to the subplot
	for i, name in enumerate(filenames):

		# Importing Network
		A = np.genfromtxt('./' + str(folder) + '/' + str(name) + '.csv', delimiter=',')

		# Printing Graph (testing)
		G = nx.Graph(A)

		# Turning matrix into an Object
		print('\n========================================================')
		print(f' --- {name} --- ')
		net = NetworkStat(A)
		print(f' Number of Vertices: {net.n} | Number of Edges: {net.m}')

		# Network Degree
		net.averageDegree()
		print(f' Average Degree: {net.d_avg}')

		# Dominant Eigenvalue
		net.domEig()
		print(f' Dominant Eigenvalue: {net.lam:0.3f}')

		# Degree Distribution
		net.degreeDis()
		deg_vector = net.degD

		# Plotting the Degree Histograms
		ax1 = fig1.add_subplot(3, 2, i + 1)
		ax1.hist(deg_vector, bins=net.n, ec='white', density=True, color=color)
		ax1.set_title(name)
		ax1.set_xlabel('Degree')

		# Plotting the Histogram of Eigenvalues
		w, v = LA.eig(A)
		ax2 = fig2.add_subplot(3, 2, i + 1)
		ax2.hist(w, color='black', ec='white', bins=net.n)
		ax2.set_title(name)
		ax2.set_xlabel('Eigenvalues')

		# Plotting the Graphs
		ax3 = fig3.add_subplot(3, 2, i + 1)
		nx.draw(G, node_color=range(A.shape[0]), ax=ax3)
		ax3.set_title(name)

	# Cleaning up figure
	fig1.subplots_adjust(hspace=0.4, wspace = 0.1)
	fig2.subplots_adjust(hspace=0.4, wspace = 0.1)

	# Saving Figures
	# fig1.savefig(f'./figures/hist_{folder}.png')
	# fig2.savefig(f'./figures/graph_{folder}.png')
	# fig3.savefig(f'./figures/eig_{folder}.png')

	plt.show()

# ========================================================
if __name__ == "__main__":
	main()