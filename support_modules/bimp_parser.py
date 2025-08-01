import xml.etree.ElementTree as ET
import json
import uuid
import datetime
from collections import defaultdict

# Define namespaces
QBP_NS = "http://www.qbp-simulator.com/Schema201212"
BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"

# Register namespaces for pretty printing
ET.register_namespace('qbp', QBP_NS)
ET.register_namespace('bpmn', BPMN_NS)

def embed_qbp_simulation(bpmn_path, resources_json_path, bpmn_bimp_path):
    """
    Embeds QBP simulation information from a JSON file into a BPMN XML file.

    Args:
        bpmn_path (str): Path to the input BPMN file.
        resources_json_path (str): Path to the JSON file containing simulation data.
        bpmn_bimp_path (str): Path where the modified BPMN file will be saved.
    """
    try:
        # Parse the existing BPMN file
        tree = ET.parse(bpmn_path)
        root = tree.getroot()

        # Load simulation data from JSON
        with open(resources_json_path, 'r', encoding='utf-8') as f:
            bimp = json.load(f)

        # Remove any existing simulation info blocks to prevent duplicates
        for elem in root.findall(f"{{{QBP_NS}}}processSimulationInfo"):
            root.remove(elem)

        # --- Create the <qbp:processSimulationInfo> block ---
        # Attributes for processSimulationInfo - using defaults as they are not in the provided JSON
        sim_info_attrs = {
            "currency": "EUR", # Default currency
            "id": f"qbp_{uuid.uuid4()}", # Generated ID
            "processInstances": "0", # Default case count
            "startDateTime": datetime.datetime.now().isoformat(timespec='microseconds') + "+00:00" # Current time
        }
        sim_info = ET.Element(f"{{{QBP_NS}}}processSimulationInfo", attrib=sim_info_attrs)

        # --- 1. Arrival time distribution ---
        arrival = bimp.get("arrival_time_distribution")
        if arrival:
            arrival_attrs = {
                "type": arrival["distribution_name"].upper() # Ensure uppercase for distribution type
            }
            # Map distribution parameters to arg1, arg2, mean based on common patterns or first available
            params = [p["value"] for p in arrival.get("distribution_params", [])]
            if len(params) > 0:
                arrival_attrs["arg1"] = str(params[0])
            if len(params) > 1:
                arrival_attrs["arg2"] = str(params[1])
            # Assuming 'mean' is the 4th parameter if available, or 0 if not
            if len(params) > 3:
                arrival_attrs["mean"] = str(params[3])
            else:
                arrival_attrs["mean"] = "0" # Default mean if not enough parameters

            arrival_node = ET.SubElement(sim_info, f"{{{QBP_NS}}}arrivalRateDistribution", attrib=arrival_attrs)
            ET.SubElement(arrival_node, f"{{{QBP_NS}}}timeUnit").text = "seconds" # Default time unit

        # --- 2. Timetables (Ensuring only ONE calendar) ---
        timetables_data = bimp.get("arrival_time_calendar", [])
        
        # Define the single primary calendar ID
        primary_calendar_id = "Discovered_CASES_ARRIVAL_CALENDAR"

        if timetables_data:
            timetables_node = ET.SubElement(sim_info, f"{{{QBP_NS}}}timetables")

            # Add the primary arrival time calendar, marked as default
            timetable_attrs = {
                "default": "true",
                "id": primary_calendar_id,
                "name": primary_calendar_id
            }
            timetable_node = ET.SubElement(timetables_node, f"{{{QBP_NS}}}timetable", attrib=timetable_attrs)
            rules_node = ET.SubElement(timetable_node, f"{{{QBP_NS}}}rules")
            for rule_data in timetables_data:
                rule_attrs = {
                    "fromTime": rule_data.get("beginTime"),
                    "fromWeekDay": rule_data.get("from"),
                    "toTime": rule_data.get("endTime"),
                    "toWeekDay": rule_data.get("to")
                }
                # Filter out None values from attributes
                rule_attrs = {k: v for k, v in rule_attrs.items() if v is not None}
                ET.SubElement(rules_node, f"{{{QBP_NS}}}rule", attrib=rule_attrs)
            
            # No other calendars are generated, adhering to the "only one calendar" rule.

        # --- 3. Resources ---
        resources_data = bimp.get("resource_profiles", [])
        if resources_data:
            resources_node = ET.SubElement(sim_info, f"{{{QBP_NS}}}resources")
            for rp in resources_data:
                for res in rp.get("resource_list", []):
                    resource_attrs = {
                        "costPerHour": str(res.get("cost_per_hour", "20")),
                        "id": res.get("id"),
                        "name": res.get("name"),
                        "timetableId": primary_calendar_id, # All resources now reference the single primary calendar
                        "totalAmount": str(res.get("amount", "1"))
                    }
                    # Filter out None values from attributes
                    resource_attrs = {k: v for k, v in resource_attrs.items() if v is not None}
                    ET.SubElement(resources_node, f"{{{QBP_NS}}}resource", attrib=resource_attrs)

        # --- 4. Tasks/Elements (durations + resource IDs) ---
        elements_node = ET.SubElement(sim_info, f"{{{QBP_NS}}}elements")
        
        # Collect all tasks and their assigned resources
        tasks_with_resources = defaultdict(list)
        for rp in bimp.get("resource_profiles", []):
            for res_info in rp.get("resource_list", []):
                resource_id = res_info.get("id")
                for task_id in res_info.get("assignedTasks", []):
                    tasks_with_resources[task_id].append(resource_id)

        # Process tasks found from assignedTasks
        for task_id, assigned_resource_ids in tasks_with_resources.items():
            # Generate a new QBP element ID
            qbp_element_id = f"qbp_{uuid.uuid4()}"
            
            el = ET.SubElement(elements_node, f"{{{QBP_NS}}}element", attrib={
                "elementId": task_id,
                "id": qbp_element_id
            })
            
            # Add a default FIXED duration distribution as duration data is not in the new JSON
            dur = ET.SubElement(el, f"{{{QBP_NS}}}durationDistribution", attrib={
                "type": "FIXED",
                "arg1": "0",
                "arg2": "0",
                "mean": "60" # Default to 60 seconds
            })
            ET.SubElement(dur, f"{{{QBP_NS}}}timeUnit").text = "seconds"
            
            # Add resource IDs
            if assigned_resource_ids:
                res_ids_node = ET.SubElement(el, f"{{{QBP_NS}}}resourceIds")
                for res_id in assigned_resource_ids:
                    ET.SubElement(res_ids_node, f"{{{QBP_NS}}}resourceId").text = res_id

        # --- 5. Branching probabilities (sequenceFlows) ---
        gw_probs = bimp.get("gateway_branching_probabilities", [])
        if gw_probs:
            seq_flows = ET.SubElement(sim_info, f"{{{QBP_NS}}}sequenceFlows")
            for gw in gw_probs:
                for path in gw.get("probabilities", []):
                    if "path_id" in path and "value" in path:
                        ET.SubElement(seq_flows, f"{{{QBP_NS}}}sequenceFlow", attrib={
                            "elementId": path["path_id"],
                            "executionProbability": str(path["value"])
                        })

        # Append the simulation section to the BPMN root
        root.append(sim_info)

        # Write the modified XML to the output file
        tree.write(bpmn_bimp_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ QBP simulation info embedded into: {bpmn_bimp_path}")

    except FileNotFoundError:
        print(f"Error: One of the files not found. BPMN: {bpmn_path}, JSON: {resources_json_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {resources_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
        
        # PATH ="data/3.bps_tobe/PurchasingExample/20250711_045405_ECFE250D_FF49_42FB_B130_CFEFEA1601C8/best_result/"
        PATH = "data/3.bps_asis/PurchasingExample/20250711_045127_FF69E080_4D30_4C7D_BAE1_874FD40E9A20/best_result/"
        embed_qbp_simulation(
           f"{PATH}/PurchasingExample.bpmn",
           f"{PATH}/PurchasingExample.json",
           f"{PATH}/PurchasingExample_bimp_version.bpmn")
