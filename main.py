import numpy as np
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
		self.den = 2 * self.m / (self.n(self.n - 1))

	def domEig(self):
		w, v = LA.eig(self.data)
		self.lam = np.amax([np.amax(w), np.amax(-w)]).real

	def degreeDis(self):
		self.degD = np.sum(self.data, axis = 1)

	def cluster(self):

		# Cardinality of the loops (loops are on diagonal)
		triangles = np.linalg.matrix_power(self.data, 3)
		diag = np.diag(triangles)

		# Clustering Vector along with coefficient
		cc_vec = 2 * diag / (self.degD * (self.degD - 1))
		self.cc_vec = cc_vec[cc_vec < 1e8] # removing infinity values
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
	for i, name in enumerate(filenames):

		# Importing Network
		A = np.genfromtxt('./' + str(folder) + '/' + str(name) + '.csv', delimiter=',')

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

		# Dominant Eigenvalue
		net.degreeDis()
		deg_vector = net.degD

		# Clustering coefficient and distribution (NOTICE: .cluster() must be run after degree distribution)
		net.cluster()
		cc_vec, cc = net.cc_vec, net.cc

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

	# Cleaning up figure
	fig1.subplots_adjust(hspace=0.4, wspace = 0.1)
	fig2.subplots_adjust(hspace=0.4, wspace = 0.1)

	fig1.savefig(f'./figures/deg_{folder}')
	fig2.savefig(f'./figures/clust_{folder}')

	# plt.show()

# ========================================================
if __name__ == "__main__":
	main()