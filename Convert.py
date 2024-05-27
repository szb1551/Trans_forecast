import openpyxl
import xml.etree.ElementTree as ET
import math


def create_node_xml(node_id, x, y):
    node_elem = ET.Element('junction')
    node_elem.set('id', str(node_id))
    node_elem.set('type', 'unregulated')
    node_elem.set('x', str(x))
    node_elem.set('y', str(y))
    node_elem.set('incLanes', '')
    node_elem.set('intLanes', '')
    node_elem.set('shape', '-0.00,-0.05,-0.00,-3.25')  # Adjust shape as per requirement
    return node_elem


def create_edge_xml(from_node_id, to_node_id, edge_id, length):
    edge_elem = ET.Element('edge')
    edge_elem.set('id', edge_id)
    edge_elem.set('from', str(from_node_id))
    edge_elem.set('to', str(to_node_id))
    edge_elem.set('priority', '-1')

    # Create lane for the edge
    lane_elem = ET.SubElement(edge_elem, 'lane')
    lane_elem.set('id', f'{edge_id}_0')  # Lane ID
    lane_elem.set('index', '0')  # Lane index
    lane_elem.set('speed', '13.90')  # Lane speed
    lane_elem.set('length', str(length))  # Lane length
    lane_elem.set('shape', f'0.00,-1.65 {length}.00,-1.65')  # Lane shape

    return edge_elem

def write_sumo_network(nodes, edges, output_file):
    root = ET.Element('net')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('version', '0.13')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/net_file.xsd')

    location_elem = ET.SubElement(root, 'location')
    location_elem.set('netOffset', '250.00,0.00')
    location_elem.set('convBoundary', '0.00,0.00,501.00,0.00')
    location_elem.set('origBoundary', '-250.00,0.00,251.00,0.00')
    location_elem.set('projParameter', '!')

    for node in nodes:
        root.append(node)

    for edge in edges:
        root.append(edge)

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def lat_lon_to_meters(lat, lon):
    center_latitude = 121.573191  # Example latitude
    center_longitude = 38.946204  # Example longitude
    # Convert latitude and longitude to meters relative to the center point
    lat_m = (lat - center_latitude) * 111000  # Approximate conversion: 1 degree of latitude ≈ 111000 meters
    lon_m = (lon - center_longitude) * 111000 * math.cos(math.radians(center_latitude))  # Corrected for longitude
    return lat_m, lon_m

def main():
    try:
        # Read latitude and longitude data
        nodes = []
        with open('data/数据源/道路信息.txt', 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                parts = line.strip().split(',')
                x = float(parts[1])
                y = float(parts[2])
                x, y = lat_lon_to_meters(x, y)
                nodes.append(create_node_xml(i, x, y))

        # Read adjacency matrix data
        workbook = openpyxl.load_workbook('data/数据源/zuobiaodaoluxinxi.xlsx')
        sheet = workbook['Sheet1']
        adjacency_matrix = []
        for row in sheet.iter_rows(values_only=True):
            row = row[1:]  # Skip first column
            adjacency_matrix.append(row)
        adjacency_matrix = adjacency_matrix[1:]
        # Create edges
        edges = []
        for i, row in enumerate(adjacency_matrix):
            for j, value in enumerate(row):
                if value == 1:
                    from_node_id = i + 1  # Excel row index starts from 1
                    to_node_id = j + 1
                    edge_id = f'{from_node_id}to{to_node_id}'  # Unique edge id
                    edges.append(create_edge_xml(from_node_id, to_node_id, edge_id, 1000))  # Temporary edge length

        # Write SUMO network file
        write_sumo_network(nodes, edges, 'network.net.xml')
        print("SUMO network file generated successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure input files exist.")

if __name__ == '__main__':
    main()
