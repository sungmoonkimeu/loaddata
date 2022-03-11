from tkinter import Tk, filedialog
import os

cwd = os.getcwd()
root = Tk()         # pointing root to Tk() to use it as Tk() in program.
root.withdraw()     # Hides small tkinter window.
root.attributes('-topmost', True)   # Opened windows will be active. above all windows despite of selection.

path_dir = filedialog.askdirectory(initialdir=cwd)     # Returns opened path as str
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

    databeginning = 11   # 4 is normal, sometimes it should be changed to 5
    fn2 = path_dir + '_edited//' + tmp[0] + '_edited.txt'
    eol = 5000           # end of line
    with open(fn) as fp:
        for i, line in enumerate(fp):
            if databeginning < i < eol+databeginning+2:
                line2 = line.replace("--", "\t")
                with open(fn2, 'a') as fp2:
                    fp2.write(line2)




