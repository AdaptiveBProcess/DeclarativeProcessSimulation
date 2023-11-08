import os
import glob
import sys
def main(argv):
    print("This is the folder")
def ReturnFolderName():
    directory='GenerativeLSTM\output_files'
    folders = glob.glob(os.path.join(directory, '*/'))
    folder = max(folders, key=os.path.getctime)
    return folder
if __name__ == "__main__":
    main(sys.argv[1:])
