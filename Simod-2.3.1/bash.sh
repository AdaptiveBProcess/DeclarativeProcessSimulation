#!/bin/bash

#Remove all the files inside Simod outputs
rm -rf outputs/*

#Receive the event log filename
logname=$1

#Execute simod for the diferent Event logs
python simod_console.py -f "$logname" -m sm3

# Source folder
source_folder="outputs"

# Destination folder to copy the .bpmn files
destination_folder="../GenerativeLSTM/input_files/simod"

# Find all .bpmn files within the source folder and copy them to the destination folder
find "$source_folder" -type f -name "*.bpmn" -exec cp {} "$destination_folder" \;