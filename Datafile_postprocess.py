"""Analysing datafiles from OFDR device.
"""

import string
import os
path_dir = 'Data3'
file_list = os.listdir(path_dir)

for nn in range(len(file_list)):
    fn = path_dir + "//" + file_list[nn]

    # test
    # fn = 'Data2//9_30deg_hibi_Upper.txt'
    # fn2 = 'Data2_edited//9_30deg_hibi_Upper2.txt'
    tmp = file_list[nn].split('.')

    fn2 = 'Data3_edited//' + tmp[0] + '_edited.txt'
    with open(fn) as fp:
        for i, line in enumerate(fp):
            if i > 11:
                print(line)  # 26th line
                with open(fn2, 'a') as fp2:
                    fp2.writelines(fp)

#mydatax = pd.read_table('Data2//9_30deg_hibi_Upper.txt')
