from glob import glob
import pandas as pd
import numpy as np
import random
    
def getCur_Res_Time(paths, dim):
    I = np.zeros((len(paths), dim))
    R = np.zeros((len(paths), dim))
    T = np.zeros((len(paths), dim))
    for i, path in enumerate(paths):
        df = pd.read_csv(path, sep=',', header=None,skiprows=4, encoding='unicode_escape')
        time = df.values[:,0]
        resistance = df.values[:,3]
        voltage = df.values[:,1]
        current = voltage / resistance
        st = len(time[time < 1200])
        time = time[st:]
        resistance = resistance[st:]
        current = current[st:]
        randIndex = random.sample(range(len(time)), dim)
        randIndex.sort()
        I[i] = np.take(current, randIndex)
        R[i] = np.take(resistance, randIndex)
        T[i] = np.take(time, randIndex)
    return I, R, T
  
nh3_path = glob('C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/WS2/NH3 10ppm/*.csv')
no2_path = glob('C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/WS2/NO2 10ppm/*.csv')
mixture_path = glob('C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/WS2/NO2 10ppm NH3 10ppm/*.csv')
mixture2_path = glob('C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/WS2/*.csv')

dimension = np.Inf
paths = nh3_path + no2_path + mixture_path + mixture2_path
print(len(paths))
# get the smallest dimension
for path in paths:
    df = pd.read_csv(path, sep=',', header=None, skiprows=4, encoding='unicode_escape')
    time = df.values[:,0]

    # resistance = df.values[:,3]
    st = len(time[time < 1200])

    time = time[st:]
    print("st:", st)
    # resistance = resistance[st:]
    if dimension > time.shape[0]:
        print("!",path)
        dimension = time.shape[0]
        print("dimen:",dimension)

# get I, R, T        
nh3I, nh3R, nh3T = getCur_Res_Time(nh3_path, dimension) 
no2I, no2R, no2T = getCur_Res_Time(no2_path, dimension) 
mixtureI, mixtureR, mixtureT = getCur_Res_Time(mixture_path, dimension) 
mixture2I, mixture2R, mixture2T = getCur_Res_Time(mixture2_path, dimension)  
          
# stack current and resistance
nh3 = np.hstack((nh3I, nh3R))
no2 = np.hstack((no2I, no2R))
mixture = np.hstack((mixtureI, mixtureR))
mixture2 = np.hstack((mixture2I, mixture2R))

# make npy 
x = np.vstack((nh3, no2, mixture, mixture2))
y = np.array([0]*len(nh3) + [1]*len(no2) + [2]*len(mixture) + [3]*len(mixture2))
np.save('C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/remake_data/modified_x', x)
np.save('C:/Users/shins/Desktop/MLPA/E-nose/code/shin_prof_code/enose_codes/codes/remake_data/modified_y', y)