import pandas as pd

# Load the CSV file into a list of lines
file_path = 'GenerativeLSTM/output_files/simulation_stats/ConsultaDataMining201618.csv'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find the start of the "Individual task statistics" section
start_idx = None
for i, line in enumerate(lines):
    if 'Individual task statistics' in line:
        start_idx = i + 1
        break

if start_idx is None:
    raise ValueError("Could not find 'Individual task statistics' section in the CSV file")

# Read the individual task statistics section into a DataFrame
task_stats = pd.read_csv(file_path, skiprows=start_idx)

# Print the columns to understand the structure
print(task_stats.columns)

summary = []

summary.append("Individual Task Statistics")
summary.append("--------------------------")

# Ensure task_stats contains the expected columns
expected_columns = 26  # Adjust this number based on the actual number of columns you expect

if task_stats.shape[1] < expected_columns:
    raise ValueError(f"DataFrame should have at least {expected_columns} columns")

for index, row in task_stats.iterrows():
    try:
        summary.append(f"Task: {row['Name']}")
        summary.append(f"- Avg duration: {float(row['Avg duration']):,.2f}")
        summary.append(f"- Min duration: {float(row['Min duration']):,.2f}")
        summary.append(f"- Max duration: {float(row['Max duration']):,.2f}")
        summary.append(f"- Total duration: {float(row['Total duration']):,.2f}")
        summary.append(f"- Avg waiting time: {float(row['Avg waiting time']):,.2f}")
        summary.append(f"- Min waiting time: {float(row['Min waiting time']):,.2f}")
        summary.append(f"- Max waiting time: {float(row['Max waiting time']):,.2f}")
        summary.append(f"- Total waiting time: {float(row['Total waiting time']):,.2f}")
        summary.append(f"- Avg idle time: {float(row['Avg idle time']):,.2f}")
        summary.append(f"- Min idle time: {float(row['Min idle time']):,.2f}")
        summary.append(f"- Max idle time: {float(row['Max idle time']):,.2f}")
        summary.append(f"- Total idle time: {float(row['Total idle time']):,.2f}")
        summary.append(f"- Avg cost: {float(row['Avg cost']):,.2f}")
        summary.append(f"- Min cost: {float(row['Min cost']):,.2f}")
        summary.append(f"- Max cost: {float(row['Max cost']):,.2f}")
        summary.append(f"- Total cost: {float(row['Total cost']):,.2f}")
        summary.append(f"- Avg cost over thresh: {float(row['Avg cost over thresh']):,.2f}")
        summary.append(f"- Min cost over thresh: {float(row['Min cost over thresh']):,.2f}")
        summary.append(f"- Max cost over thresh: {float(row['Max cost over thresh']):,.2f}")
        summary.append(f"- Total cost over thresh: {float(row['Total cost over thresh']):,.2f}")
        summary.append(f"- Avg duration over thresh: {float(row['Avg duration over thresh']):,.2f}")
        summary.append(f"- Min duration over thresh: {float(row['Min duration over thresh']):,.2f}")
        summary.append(f"- Max duration over thresh: {float(row['Max duration over thresh']):,.2f}")
        summary.append(f"- Total duration over thresh: {float(row['Total duration over thresh']):,.2f}")
        summary.append(f"- Count: {int(row['Count'])}")
        summary.append("")
    except (KeyError, ValueError) as e:
        summary.append(f"Error processing row {index}: {e}")

# Save summary to a text file
with open('summary.txt', 'w') as file:
    for line in summary:
        file.write(line + '\n')
        
        