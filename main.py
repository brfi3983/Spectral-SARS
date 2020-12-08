import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from SIR_epidemic import *

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

	# figures
	fig1 = plt.figure()
	fig2, ax2 = plt.subplots(3,2)
	# fig3, ax3 = plt.subplots(3,2)

	# figure titles
	fig1.suptitle(title, fontsize=30)
	fig2.suptitle('test', fontsize=30)

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

		# Plotting the Histograms
		ax1 = fig1.add_subplot(3, 2, i + 1)
		ax1.hist(deg_vector, bins=net.n, ec='white', density=True, color=color)
		ax1.set_title(name)
		ax1.set_xlabel('Degree')

		## SIR Model for Matrix A (100 simulations)
		model = SIR_class(A)

		k_arr = [1, 2, 3, 4, 5]
		B = 4e-4
		T = 1e3

		sims_R = {}
		p0 = {}
		for k in k_arr:
			sim_R = np.empty(0)
			mu = 100 * B / k
			del_t = 1e-3 /B
			for i in range(100):
				[S, I, R] = model.SIR(B, mu, del_t, T, vaccinated=0)
				sim_R = np.append(sim_R, R)

			sims_R[str(k)] = sim_R
			p0[str(k)] = B * net.d_avg / mu

			# Plotting distribution of simulations for each k (for each dataset - CREATE NEW FIGURE or overlay histograms...)
			ax2 = fig2.add_subplot(3, 2, k + 1)
			# ax3 = fig3.add_subplot(3, 2, k + 1)
			# ax2.hist(sims[str(k)], bins=net.n, ec='white')
			ax2.set_title(r'$p_0$ value of ' + str(p0[str(k)]))
			# ax2.set_xlabel('Node in Graph')
			# ax2 = fig2.add_axes
			ax2.hist(sims_R[str(k)], bins=15, ec='white')

		break
		# print(p0['1'])
		# print(sims['1'])
	# Cleaning up figure
	fig1.subplots_adjust(hspace=0.4, wspace = 0.1)
	# ax2.legend()
	fig2.subplots_adjust(hspace=0.4, wspace = 0.1)
	plt.show()

# ========================================================
if __name__ == "__main__":
	main()