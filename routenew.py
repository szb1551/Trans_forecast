import xml.etree.ElementTree as ET
import random


# Parse the network file to extract edge IDs and connections
def extract_edges_and_connections(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()
    edges = {edge.get('id'): [conn.get('to') for conn in edge.findall('./connection')] for edge in
             root.findall('.//edge')}
    return edges


# Perform depth-first traversal to generate routes
def depth_first_traversal(edges, start_edge, visited, min_nodes):
    route = [start_edge]
    visited.add(start_edge)
    if len(route) >= min_nodes:
        return route
    next_edges = edges.get(start_edge)
    if next_edges:
        for next_edge in next_edges:
            if next_edge not in visited:
                new_route = depth_first_traversal(edges, next_edge, visited, min_nodes)
                if new_route:
                    route += new_route
                    if len(route) >= min_nodes:
                        break
    return route if len(route) >= min_nodes else None


# Generate routes for vehicles
def generate_routes(edges, num_routes, min_nodes):
    routes = []
    visited_edges = set()
    while len(routes) < num_routes:
        start_edge = random.choice(list(edges.keys()))
        if start_edge not in visited_edges:
            visited_edges.add(start_edge)
            route = depth_first_traversal(edges, start_edge, set(), min_nodes)
            if route:
                route_elem = ET.Element('route')
                route_elem.set('id', f'route{len(routes)}')
                route_elem.set('edges', ' '.join(route))
                routes.append(route_elem)
    return routes


# Generate vehicles using the routes
def generate_vehicles(routes, num_vehicles_per_route):
    vehicles = []
    for i, route_elem in enumerate(routes):
        for j in range(num_vehicles_per_route):
            vehicle_elem = ET.Element('vehicle')
            vehicle_elem.set('depart', '1')  # Depart time (in seconds)
            vehicle_elem.set('id', f'veh{i}_{j}')
            vehicle_elem.set('route', route_elem.get('id'))
            vehicle_elem.set('type', 'Car')  # Vehicle type
            vehicles.append(vehicle_elem)
    return vehicles


# Write routes and vehicles to rou.xml file
def write_rou_file(routes, vehicles, output_file):
    root = ET.Element('routes')
    root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
    vtype_elem = ET.Element('vType')
    vtype_elem.set('accel', '1.0')
    vtype_elem.set('decel', '5.0')
    vtype_elem.set('id', 'Car')
    vtype_elem.set('length', '5.0')
    vtype_elem.set('minGap', '2.0')
    vtype_elem.set('maxSpeed', '50.0')
    vtype_elem.set('sigma', '0')
    root.append(vtype_elem)
    for route_elem in routes:
        root.append(route_elem)

    for vehicle_elem in vehicles:
        root.append(vehicle_elem)

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def main():
    network_file = 'network.net.xml'  # Path to the network file
    output_file = 'vehicle_routes.rou.xml'  # Output rou.xml file
    num_routes = 5  # Number of routes to generate
    num_vehicles_per_route = 3  # Number of vehicles per route
    min_nodes = 1  # Minimum number of nodes in a route

    # Extract edge IDs and connections from the network file
    edges = extract_edges_and_connections(network_file)

    # Generate routes based on edges and connections
    routes = generate_routes(edges, num_routes, min_nodes)

    # Generate vehicles using the routes
    vehicles = generate_vehicles(routes, num_vehicles_per_route)

    # Write routes and vehicles to rou.xml file
    write_rou_file(routes, vehicles, output_file)
    print("rou.xml file generated successfully.")


if __name__ == '__main__':
    main()
