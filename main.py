# Useful packages
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import shutil
import random
import timeit
import yaml
from skimage import data
from skimage.feature import blob_dog

# Defining class for XY model
class XY():

	# Constructor for XY model
	def __init__(self, dims, T, save, pad):

		# Check run number
		self.runNum = len(glob.glob('runs/run*')) + 1

		# Defining root folder
		self.root = f'runs/run{str(self.runNum).zfill(3)}/'

		# Save run every frame?
		self.save = save

		# How much padding for frames
		self.pad = pad

		# dims: (W, H)
		self.dims = dims

		# Total number of lattice points
		N = dims[0]*dims[1]
		self.N = N

		# Width of lattice
		W = dims[0]

		# Defining nearest neighbours of a point
		self.nns = {i: [(i//W)*W + (i-1)%W, (i-W)%N,
				(i//W)*W + (i+1)%W, (i+W)%N]\
				for i in range(N)}

		# Defining winding neighbours in order
		self.wns = {i: [i+1, i+W+1, i+W, i+W-1, i-1, i-W-1, i-W, i-W+1]\
				for i in range(N)}

		# Defining randomly oriented spins
		self.angles = np.random.random(self.N)*(2*np.pi)

		# Defining temperature of model
		self.T = T

		# Reduced energy
		self.RedE = self.energy()/self.N

		# Keep track of current frame
		self.frame = 0

		return

	# Set up experiment
	def expSetup(self, expYAML):

		# Create root folder to store experiments
		os.mkdir(self.root)

		# Copy over config file
		shutil.copyfile(expYAML, self.root+expYAML.split('/')[-1])

		# Creating folder for orientations
		os.mkdir(self.root + 'orientations/')		
		# Creating folder for orientations images
		os.mkdir(self.root + 'orientationsFrames/')
		# Creating folder for schlieren textures
		os.mkdir(self.root + 'schlieren/')
		# Creating folder for schlieren textures images
		os.mkdir(self.root + 'schlierenFrames/')
		# Creating folder for energy densities
		os.mkdir(self.root + 'energy/')
		# Creating folder for energy density images
		os.mkdir(self.root + 'energyFrames/')
		# Creating folder for detected defects
		os.mkdir(self.root + 'defects/')
		# Creating folder for detected defects images
		os.mkdir(self.root + 'defectsFrames/')


	# Elastic energy stored in model
	def energy(self):

		# Variable to store energy
		E = 0

		# Go through every lattice point and add up energy
		for i in range(self.N):
			E = E - sum(np.cos(self.angles[i] - self.angles[j]) for j in self.nns[i])

		# Return energy
		return E

	# Plot energy density 
	def plotEnergyDensity(self):

		# Clear plots
		plt.clf()

		# Array to store energies
		energies = np.zeros(self.N)

		# Go through each lattice point and obtain energy density
		for i in range(self.N):

			energies[i] = -sum(np.cos(self.angles[i] - self.angles[j]) for j in self.nns[i])

		# Reshape array to a grid
		energies = energies.reshape(self.dims)

		# Store energy density
		np.savetxt(self.root + f'energy/{str(self.frame).zfill(self.pad)}.txt', energies)

		# Create and save plot
		plt.imshow(energies, cmap='viridis')
		plt.xlim(-0.5, self.dims[0]-0.5)
		plt.ylim(-0.5, self.dims[1]-0.5)
		plt.colorbar()
		plt.savefig(self.root + f'energyFrames/{str(self.frame).zfill(self.pad)}.png', dpi=300)
		# plt.show()

	# Create schlieren texture of model
	def schlieren(self):

		# Clear plots
		plt.clf()

		# Compute texture with crossed polarizers
		schl = np.sin(2*self.angles)**2

		# Reshape texture to grid
		schl = schl.reshape(self.dims)

		# Retain schlieren texture
		self.schl = schl

		# Store schlieren texture
		np.savetxt(self.root + f'schlieren/{str(self.frame).zfill(self.pad)}.txt', schl)

		# Plot and save schlieren texture
		plt.imshow(schl, cmap='gray')
		plt.xlim(-0.5, self.dims[0]-0.5)
		plt.ylim(-0.5, self.dims[1]-0.5)
		plt.colorbar()
		plt.savefig(self.root + f'schlierenFrames/{str(self.frame).zfill(self.pad)}.png', dpi=300)

	# Heat up film
	def heat(self, T):
		self.T = T

	# Take a time step
	def step(self):

		# Create array of indices
		indices = np.arange(self.N)

		# Shuffle indices
		random.shuffle(indices)

		# Pick an index at random
		for i in indices:

			# Find initial energy
			E_initial = -sum(np.cos(self.angles[i] - self.angles[j]) for j in self.nns[i])

			# Take random change in orientation
			dTheta = np.random.uniform(-np.pi, np.pi)

			# Find changed angle
			fAngle = self.angles[i] + dTheta

			# Find changed energy
			E_final = -sum(np.cos(fAngle - self.angles[j]) for j in self.nns[i])

			# Change in energy
			dE = E_final - E_initial

			# Check if energetically favourable, change if favourable

			# If film is completely cold, change angle
			if self.T == 0:
				self.angles[i] = fAngle
			# Use taylor series for small ratio
			elif dE/self.T < 0.005:
				if np.random.uniform(0, 1) < 1 - dE/self.T:
					self.angles[i] = fAngle
			# Use exact form for large ratio
			elif np.random.uniform(0, 1) < np.exp(-dE/self.T):
				self.angles[i] = fAngle

		# Reset orientations to between 0 and 2pi
		self.angles = np.mod(self.angles, 2*np.pi)

		# Compute reduced energy
		self.RedE = self.energy()/self.N

		# Iterate frame
		self.frame = self.frame + 1

	# Evolve the film
	def evolve(self, nSteps=int(1e2), save=False, saveRate=1):

		# Go through each frame
		for i in range(nSteps):

			# Display progress every 10 frames
			if i%10 == 0:
				print(f'{100*i/nSteps:.2f}%')

			# Check if user wants to save and fits saverate
			if i%saveRate == 0 and self.save:
				self.display()
				self.plotEnergyDensity()
				self.schlieren()
				self.detectDefects()

			# Take a step
			self.step()

		# Check if the user wants to save
		if self.save:
			self.display()
			self.plotEnergyDensity()
			self.schlieren()
			self.detectDefects()

	# Displaying plot
	def display(self):

		# Clear plots
		plt.clf()

		# Get colour map
		colormap = plt.cm.get_cmap('hsv')

		# Reshape spins
		gAngles = self.angles.view()
		gAngles = gAngles.reshape(self.dims)

		# Define a colour map corresponding to orientation
		rotations = self.angles/(2*np.pi)
		colorImage = colormap(gAngles/(2*np.pi))
		sm = plt.cm.ScalarMappable(cmap=colormap)
		sm.set_clim(vmin=0, vmax=2*np.pi)

		# Save orientations
		np.savetxt(self.root + f'orientations/{str(self.frame).zfill(self.pad)}.txt', gAngles)

		# Display and save plot
		plt.imshow(colorImage)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.xlim(-0.5, self.dims[0]-0.5)
		plt.ylim(-0.5, self.dims[1]-0.5)
		plt.title(f'$k_BT/J = {self.T}$'+f'$, t={(self.frame//10)/10}~s$')
		plt.axis("off")
		plt.colorbar(sm).set_label('orientation')
		plt.savefig(self.root + f'orientationsFrames/{str(self.frame).zfill(self.pad)}.png', dpi=300)

	# Find defects using blob detection of energy density
	def detectDefects(self):

		# Max radius of a defect blob
		maxRad = 2

		# Clear plots
		plt.clf()

		# Load energy density values
		density = np.loadtxt(self.root + f'energy/{str(self.frame).zfill(self.pad)}.txt')

		# Use difference of gaussian method to get blobs
		blobs = blob_dog(density, min_sigma=1, max_sigma=3, threshold=.3, overlap=0.9)

		# Update blobs to have radius in final column
		blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

		# Update blobs to swap x and y
		blobs[:, [0, 1]] = blobs[:, [1, 0]]

		# Remove blobs too large
		blobs = blobs[blobs[:, 2] < maxRad]

		# Save blobs to txt file
		np.savetxt(self.root + f'defects/{str(self.frame).zfill(self.pad)}.txt', blobs)

		# Display plot
		fig, ax = plt.subplots()
		im = ax.imshow(density)
		for blob in blobs:
			x, y, r = blob
			c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
			ax.add_patch(c)
		plt.xlim(-0.5, self.dims[0]-0.5)
		plt.ylim(-0.5, self.dims[1]-0.5)
		plt.colorbar(im)
		# plt.show()
		plt.savefig(self.root + f'defectsFrames/{str(self.frame).zfill(self.pad)}.png', dpi=300)
		plt.close('all')

# Main functioning of script
def main(expYAML):

	# Open the specified experiment yaml
	with open(expYAML, 'r') as stream:
		# Try to loadparse the yaml
		try:
			# Save it as experiment data
			expData = yaml.safe_load(stream)
		# Display an error
		except yaml.YAMLError as exc:
			# Print the exception
			print(exc)

	# Obtain padding for frames
	pad = len(str(expData['n']))

	# Create model
	model = XY(expData['dims'], float(expData['T']), expData['save'], pad)

	# If user wants to save
	if expData['save']:
		# Create run
		model.expSetup(expYAML)

	# Evolve model
	model.evolve(int(float(expData['n'])), save=expData['save'], saveRate=expData['saveRate'])

# If file is run directly
if __name__ == '__main__':

	# Define the experiment config file
	# expYAML = 'experiments/defaultExp.yml'
	expYAML = 'experiments/exp128x128_T_1E-3.yml'

	# Run main function of file
	main(expYAML)