import os
import glob
import sys
def main(argv):
    print("This is the folder")
    ReturnFolderName()
def ReturnFolderName():
    directory='GenerativeLSTM\output_files'
    folders = glob.glob(os.path.join(directory, '*/'))
    most_recent_folder = max(folders, key=os.path.getctime)
    folder = print(os.path.basename(os.path.normpath(most_recent_folder)))
    return folder
if __name__ == "__main__":
    main(sys.argv[1:])
