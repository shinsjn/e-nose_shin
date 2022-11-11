import numpy as np
import pandas as pd
import os
num_count = 0
root_file_path='C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/WS2'
root_file_list = os.listdir(root_file_path)
label=0         #lable num
result = []
final_fd = pd.DataFrame(columns=['Voltgae [V]','Current [A]','Resistance [Ω]', 'Power [W]'])
for file_list in root_file_list:
    class_file_path = root_file_path+'/'+file_list
    class_file_list = os.listdir(class_file_path)

    for file_name in class_file_list:
        file_path = class_file_path+'/'+file_name
        df = pd.read_csv(file_path,encoding='CP949', skiprows=[0, 1, 3], usecols=[1, 2, 3, 4])

        #pandas data
        df['label'] = label
        final_fd = pd.concat([final_fd,df])
        num_count = num_count + 1

        '''
        --- numpy data ---
        extract_df = df.loc[4001:6000]          #row extract to make shape 8001
        extract_df2numpy = extract_df.to_numpy()
        extract_df2numpy=extract_df2numpy.flatten()
        extract_df2numpy = np.append(extract_df2numpy,np.array(label))
        print(extract_df2numpy.shape)
        print(file_path)
        result.append(extract_df2numpy)
        '''
    label=label+1

root_file_path='C:/Users/shins/Desktop/MLPA/E-nose/data/row_data/New_NO2 NH3 mixed gas'
root_file_list = os.listdir(root_file_path)
result = []
for file_list in root_file_list:
    class_file_path = root_file_path+'/'+file_list
    class_file_list = os.listdir(class_file_path)

    for file_name in class_file_list:
        file_path = class_file_path+'/'+file_name
        df = pd.read_csv(file_path,encoding='CP949', skiprows=[0, 1, 3], usecols=[1, 2, 3, 4])

        #pandas data
        df['label'] = label
        final_fd = pd.concat([final_fd,df])
        num_count = num_count + 1

        '''
        --- numpy data ---
        extract_df = df.loc[4001:6000]          #row extract to make shape 8001
        extract_df2numpy = extract_df.to_numpy()
        extract_df2numpy=extract_df2numpy.flatten()
        extract_df2numpy = np.append(extract_df2numpy,np.array(label))
        print(extract_df2numpy.shape)
        print(file_path)
        result.append(extract_df2numpy)
        '''
    label=label+1



#save pandas data
print(num_count)
final_fd.to_csv('concat_pd_data.csv')
print(final_fd.head)
'''
#save numpy data
print(result)
result = np.array(result)
print(result.shape)
np.savetxt('sample.dat',result)
'''
