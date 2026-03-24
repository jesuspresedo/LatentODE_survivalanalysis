import glob
import numpy as np
import pandas as pd

features = [0, 1, 8, 22, 30, 59, 70, 97, 220, 439, 474, 476]
        
files = glob.glob("/home/suso/Documentos/attention/database/*.csv")
f = pd.read_csv(files[0])
columnas = f.columns
        
columnas = columnas[features]
        #columnas = np.append(columnas, 'FA')

tiempos = np.array([])		
files = sorted(files)
for filename in files:
	record_id = filename.split('/')[-1].split('.')[0]
	print(record_id)
	f = pd.read_csv(filename)
	labels = f['FA']
	tt = f['tiempo']
	longitud = tt.size
	
	for i in range(longitud):
		if labels[i] == 1:
			break
	up = i
	tiempo = tt[i-1]
	for i in range(longitud):
		if tiempo - tt[i] <= 365:
			break
	down = i
	up = up + 1
	ecgs = list(range(down,up))
	tt = tt[ecgs]
	tt = np.array(tt)
	tt = tt - tt[0]
	while(True):
		if(len(np.unique(np.concatenate((tiempos, tt)))) == len(tiempos) + len(tt)):
			tiempos = np.concatenate((tiempos,tt))
			break
		tt = tt + 1
		
tiempos = np.sort(tiempos)

