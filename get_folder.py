import os
import glob
import sys
import fileinput
def main(argv):
    print("Changing folder preferences")
    ReturnFolderName()
    #ChangeModelNameTraining()
def ChangeModelNameTraining():
    model='PurchasingExample.xes'
    #print(model)
    file_to_modify = 'dg_training.py'
    line_number=43
    with fileinput.FileInput(file_to_modify, inplace=True) as file:
            for i, line in enumerate(file, start=1):
                if i == line_number:
                    print(f"        parameters['file_name'] = '{model}'")
                else:
                    print(line, end='')       
def ChangeModelName():
    model='PurchasingExample.xes'
    print(model)
    file_to_modify = 'dg_prediction.py'
    line_number=139
    with fileinput.FileInput(file_to_modify, inplace=True) as file:
            for i, line in enumerate(file, start=1):
                if i == line_number:
                    print(f"    parameters['filename'] = '{model}'")
                else:
                    print(line, end='')           
def ReturnFolderName():
    directory='GenerativeLSTM\output_files'
    folders = glob.glob(os.path.join(directory, '*/'))
    most_recent_folder = max(folders, key=os.path.getctime)
    print(os.path.basename(os.path.normpath(most_recent_folder)))
    folder = os.path.basename(os.path.normpath(most_recent_folder))
    file_to_modify = 'dg_prediction.py'
    line_number=158
    with fileinput.FileInput(file_to_modify, inplace=True) as file:
        for i, line in enumerate(file, start=1):
            if i == line_number:
                print(f"        parameters['folder'] = '{folder}'")
            else:
                print(line, end='')
if __name__ == "__main__":
    main(sys.argv[1:])
