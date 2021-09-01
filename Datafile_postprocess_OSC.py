"""Analysing datafiles from Oscilloscope device.
"""

import string
import os

#path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//Const_acc_OSC2'
path_dir = os.getcwd() + '//Data_Vib_1_(Oscillo_Polarimeter)//010921 Vibration in second part_OSC//3.5V input'
file_list = os.listdir(path_dir)


def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory')


target_path = path_dir + "_edited"
createfolder(target_path)

for nn in range(len(file_list)):
    fn = path_dir + "//" + file_list[nn]
    tmp = file_list[nn].split('.')

    fn2 = target_path + '//' + tmp[0] + '_edited.txt'
    with open(fn) as fp:
        for i, line in enumerate(fp):
            if i >= 0:
                print(line)  # 26th line
                with open(fn2, 'a') as fp2:
                    fp2.writelines(fp)

# mydatax = pd.read_table('Data2//9_30deg_hibi_Upper.txt')
