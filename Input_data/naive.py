import xml.etree.ElementTree as ET
from datetime import datetime

# Parse the XML file
tree = ET.parse('ConsultaDataMining201618.xes')
root = tree.getroot()

# Iterate over all <date> elements and convert the timestamp to tz-naive
for date_elem in root.findall(".//date"):
    timestamp = date_elem.get('value')
    # Parse the timestamp
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Convert to tz-naive by removing the timezone info
    dt_naive = dt.replace(tzinfo=None)
    # Update the value attribute with the tz-naive timestamp
    date_elem.set('value', dt_naive.isoformat())

# Save the modified XML back to a file
tree.write('ConsultaDataMining201618_tz_naive.xes')