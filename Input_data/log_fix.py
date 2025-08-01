import xml.etree.ElementTree as ET
from datetime import datetime
import pytz

# Parse the XML file
tree = ET.parse('ConsultaDataMining201618.xes')
root = tree.getroot()

# Define the timezone you want to use
timezone = pytz.timezone('UTC')

# Iterate over all <date> elements and convert the timestamp to tz-aware
for date_elem in root.findall(".//date"):
    timestamp = date_elem.get('value')
    # Parse the timestamp
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Localize the datetime to the specified timezone
    dt_aware = timezone.localize(dt)
    # Update the value attribute with the tz-aware timestamp
    date_elem.set('value', dt_aware.isoformat())

# Save the modified XML back to a file
tree.write('ConsultaDataMining201618_tz_aware.xes')