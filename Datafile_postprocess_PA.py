"""Analysing datafiles from polarimeter POD-101

"""

import string
import os

#foldername = 'Const_Freq_Polarimeter'
#foldername = '/010921_HIBI_Const_Disp_Polarimeter/RHC'
#foldername = '/Data_Vib_2_(Hibi_losen_fasten)/1_RHC_Fasten'
#foldername = '/Data_Vib_3_(Hibi_loosen_fasten)/2_LP45'
#foldername = '/Data_Vib_4(Hibi_loosen)/Polscan_1504'
#foldername = '/Data_Vib_4(Hibi_fasten)/Pol_Scan_Tighten_1504'
foldername = '/Data_Vib_4(Lobi)/Pol_Scan_Lobi_1_1504'
#foldername = '/Data_Vib_4(Lobi)/Pol_Scan_3basis_1504'

#foldername = '/Data_Vib_1_(Oscillo_Polarimeter)/Const_volt_RHC_Polarimeter'
#path_dir = os.getcwd() + '/Data_Vib_1_(Oscillo_Polarimeter)/' + foldername


path_dir = os.getcwd() + foldername
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
    # Removing file header
    # Data starting from n th line

    databeginning = 4   # 4 is normal, sometimes it should be changed to 5
    fn2 = path_dir + '_edited//' + tmp[0] + '_edited.txt'
    with open(fn) as fp:
        for i, line in enumerate(fp):
            if i > databeginning:
                print(line)  # 26th line
                with open(fn2, 'a') as fp2:
                    fp2.writelines(fp)

#mydatax = pd.read_table('Data2//9_30deg_hibi_Upper.txt')
