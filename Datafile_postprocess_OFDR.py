"""Analysing datafiles from OFDR device.
"""

import string
import os
#path_dir = os.getcwd() + '//Data_Twsiting_(OFDR)//Data_1308//spunfiber//2nd'
path_dir = os.getcwd() + '//Data_OFDR_APC//OFDR RL measurement (220506)'
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

    fn2 = path_dir + '_edited//' + tmp[0] + '_edited.txt'
    with open(fn) as fp:
        for i, line in enumerate(fp):
            if i > 11:
                print(line)  # 26th line
                with open(fn2, 'a') as fp2:
                    fp2.writelines(fp)

