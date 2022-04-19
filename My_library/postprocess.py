from tkinter import Tk, filedialog
import os
#import numpy as np


def createfolder(directory):
    if os.path.exists(directory):
        for file in os.scandir(directory):
            os.remove(file.path)
        print(" files in the ", directory, " has been deleted.")
    else:
        os.makedirs(directory)


def postprocess_pa():
    cwd = os.getcwd()
    rootdirectory = os.path.dirname(os.getcwd())
    root = Tk()  # pointing root to Tk() to use it as Tk() in program.
    root.withdraw()  # Hides small tkinter window.
    root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite of selection.
    path_dir = filedialog.askdirectory(initialdir=rootdirectory)  # Returns opened path as str

    target_path = path_dir + "_edited"
    createfolder(target_path)

    file_list = os.listdir(path_dir)
    try:
        file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0][2:]))
    except:
        file_list = sorted(file_list, key=lambda x: int(os.path.splitext(x)[0][0:1]))

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
                    if nn == 0:
                        print(line)  # 26th line
                    with open(fn2, 'a') as fp2:
                        fp2.writelines(fp)

if __name__ == '__main__':
    postprocess_pa()