import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import linalg as LA
plt.style.use('ggplot')

# Importing Networks
pres_filenames = ['A_pres_InVS13', 'A_pres_InVS15', 'A_pres_LH10', 'A_pres_LyonSchool', 'A_pres_SFHH', 'A_pres_Thiers13']
cont_filenames = ['A_lnVS13', 'A_lnVS15', 'A_LH10', 'A_LyonSchool', 'A_SFHH', 'A_Thiers13']

# Choosing Contact vs. Presence Network
contact = True

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

	fig1 = plt.figure()
	fig2 = plt.figure()
	fig3 = plt.figure()

	fig1.suptitle(title, fontsize=30)
	fig2.suptitle(f'Graph\'s for {title}', fontsize=30)
	fig3.suptitle(f'Eigenvalue Histograms', fontsize=30)

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

		# Plotting the Histograms
		ax1 = fig1.add_subplot(3, 2, i + 1)
		ax1.hist(deg_vector, bins=net.n, ec='white', density=True, color=color)
		ax1.set_title(name)
		ax1.set_xlabel('Degree')

		# Plotting the Graphs
		ax2 = fig2.add_subplot(3, 2, i + 1)
		nx.draw(G, node_color=range(A.shape[0]), ax=ax2)
		ax2.set_title(name)

		# Plotting the Histogram of Eigenvalues
		w, v = LA.eig(A)
		ax3 = fig3.add_subplot(3, 2, i + 1)
		ax3.hist(w, color='black', ec='white', bins=net.n)
		ax3.set_title(name)
		ax3.set_xlabel('Eigenvalues')

	# Cleaning up figure
	fig1.subplots_adjust(hspace=0.4, wspace = 0.1)
	fig3.subplots_adjust(hspace=0.4, wspace = 0.1)
	plt.show()

# ========================================================
if __name__ == "__main__":
	main()