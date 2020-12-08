import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from SIR_epidemic import *

plt.style.use('ggplot')

# Importing Networks
pres_filenames = ['A_pres_InVS13', 'A_pres_InVS15', 'A_pres_LH10', 'A_pres_LyonSchool', 'A_pres_SFHH', 'A_pres_Thiers13']
cont_filenames = ['A_lnVS13', 'A_lnVS15', 'A_LH10', 'A_LyonSchool', 'A_SFHH', 'A_Thiers13']

# Choosing Contact vs. Presence Network AND part (2 = #5 and 3 = #6 in problem set)
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

	# Fraction above threshold
	fig3, ax3 = plt.subplots(3, 2, figsize=(16, 9))
	fig3.suptitle(title, fontsize=30)
	ax3 = ax3.ravel()  # changed it to vector for programming ease

	# Average # recovered above threshold
	fig4, ax4 = plt.subplots(3,2, figsize=(16,9))
	fig4.suptitle(title, fontsize=30)
	ax4 = ax4.ravel() # same as above

	# Cycle through the networks
	for i, name in enumerate(filenames):

		# Importing Network
		A = np.genfromtxt('./' + str(folder) + '/' + str(name) + '.csv', delimiter=',')

		# Turning matrix into an Object
		print('\n========================================================')
		print(f' --- {name} --- ')

		# Get basic Network Statistics
		net = NetworkStat(A)
		net.averageDegree()

		# SIR Model for Matrix A (100 simulations)
		model = SIR_class(A)

		# Constants and Iterables
		k_arr = [1, 2, 3, 4, 5]
		B = 4e-4
		T = 1e3
		nr = 0.2  #fraction threshold

		# Dictionaries and Arrays to store data
		sims_R = {}
		p0 = {}
		frac_nr = []
		avg_nr = []

		# Loop over possible k values
		fig_k = plt.figure(figsize=(16, 9))
		fig_k.suptitle(name, fontsize=30)

		# Loop through possible k (and thus p0) values...
		for k in k_arr:

			sim_R = np.empty(0) # Array to store the 100 simulations
			mu = 100 * B / k
			del_t = 1e-3 / B

			# Run simulation 100 times
			print(f'Running Simulations... (k={k})')
			for j in range(100):
				[S, I, R] = model.SIR(B, mu, T, vaccinated=False)
				sim_R = np.append(sim_R, R)

			# Store Results
			sims_R[str(k)] = sim_R
			p0[str(k)] = B * net.d_avg / mu
			frac_nr.append(sum(sim_R / net.n > nr) / 100)
			avg_nr.append(np.mean(sim_R[sim_R / net.n > nr]))

			# Plotting distribution of simulations for each k (#4 in problem set)
			ax2 = fig_k.add_subplot(3, 2, k)
			ax2.set_title(r'$p_0$ value of ' + str("%.2f" % round(p0[str(k)],2)))
			ax2.hist(sims_R[str(k)], bins=15, ec='white', color='dimgrey')

		# Adjusting figure k and creating a list from p0 dictionary
		fig_k.subplots_adjust(hspace=0.4, wspace=0.1)
		p0_array = list(p0.values())

		# Plotting percentage nodes recovered (above threshold)
		ax3[i].set_title(name)
		ax3[i].set_xlabel(r'$p_0$')
		ax3[i].set_ylabel('Percentage')
		ax3[i].fill_between(p0_array, 0, frac_nr, facecolor='seagreen', alpha=0.6)
		ax3[i].plot(p0_array, frac_nr, color='black', ls='--')

		# Plotting average number of nodes recovered (above threshold)
		ax4[i].set_title(name)
		ax4[i].set_xlabel(r'$p_0$')
		ax4[i].set_ylabel('Avg. # of Recovered')
		ax4[i].fill_between(p0_array, 0, avg_nr, facecolor='tan', alpha=0.8)
		ax4[i].plot(p0_array, avg_nr, color='black', ls='--')

		fig_k.savefig(f'./figures/{name}_p0_dist.png')
		print('Done with dataset.')

	# Cleaning up figure
	fig3.subplots_adjust(hspace=0.4, wspace=0.2)
	fig4.subplots_adjust(hspace=0.4, wspace=0.2)

	# Saving Figures
	fig3.savefig('./figures/Percentage.png')
	fig4.savefig('./figures/AverageNumRecovered.png')

	# plt.show()

# ========================================================
if __name__ == "__main__":
	main()