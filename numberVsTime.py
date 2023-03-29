import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import glob

def process(runNum):

	path = f'runs/run{str(runNum).zfill(3)}/'

	detectionsPaths = glob.glob(path+'defects/*.txt')

	array = []

	for i, dpath in enumerate(detectionsPaths):

		if i%100 == 0:

			print(f'{100*(i+1)/len(detectionsPaths):.3f}%')

		frame = int(dpath.split('\\')[-1].split('.')[0])

		mat = np.loadtxt(dpath)

		num = len(mat)

		array.append(np.array([frame, num]))

	array = np.asarray(array).T

	df = {
		'iterations': array[0],
		'N': array[1],
	}

	df = pd.DataFrame.from_dict(df)

	df.sort_values(by=['iterations'], inplace = True)

	df.to_csv(path+'number_vs_time.csv', index=False)

def fits(runNum):

	path = f'runs/run{str(runNum).zfill(3)}/'

	dfPath = path + 'number_vs_time.csv'

	df = pd.read_csv(dfPath)

	ts = df['iterations'].to_numpy().astype(np.float32)
	Ns = df['N'].to_numpy().astype(np.float32)

	mask = (ts > 100)*(ts < 100000)

	tsSelect = ts[mask]
	NsSelect = Ns[mask]

	def diffuseModel(t, a):

		return a*t**(-1)

	def JangModel(t, a):

		return a*t**(-0.9)

	def YurkeModel(N, a, b):

		return 1/(a * N * np.log(N/b))

	def anySlope(t, a, v):

		return a*t**(-v)

	# popt1, pcov1 = curve_fit(diffuseModel, tsSelect, NsSelect)

	# NsDiffuse = diffuseModel(ts, *popt1)

	# popt2, pcov2 = curve_fit(JangModel, tsSelect, NsSelect)

	# NsJang = JangModel(ts, *popt2)

	popt3, pcov3 = curve_fit(YurkeModel, NsSelect, tsSelect)

	tsYurke = YurkeModel(Ns, *popt3)

	popt4, pcov4 = curve_fit(anySlope, tsSelect, NsSelect)

	NsAny = anySlope(ts, *popt4)

	# fits = np.asarray([[*popt1, 0], [*popt2, 0], [*popt3]]).T

	# np.savetxt(path+'fits.txt', fits)
	
	plt.scatter(ts, Ns, s=0.5, color='k', label='data')
	# plt.scatter(tsSelect, NsSelect, s=1, color='r', label='data')

	# plt.plot(ts, NsDiffuse, color='g', label='diffusive')

	# plt.plot(ts, NsJang, color='r', label=r'$\nu = 0.9$')

	plt.plot(tsYurke, Ns, color='b', label='Yurke')

	plt.plot(ts, NsAny, color='r', label=f'v = {popt4[-1]:.2f}')

	plt.gca().set_yscale('log')
	plt.gca().set_xscale('log')

	# plt.xlim((5e-6, 2e-2))
	# plt.ylim((1, 1800))

	plt.gca().set_aspect('equal', adjustable='box')

	plt.legend()

	plt.show()

if __name__ == '__main__':
	fits(2)