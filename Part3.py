import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    
    # Return distance in kilometers
    return r * c

def calculate_distance_matrix(towers_dict):
    """
    Calculate distances between all pairs of towers and return a distance matrix
    
    Args:
        towers_dict: Dictionary with call signs as keys and (lat, long) tuples as values
        
    Returns:
        distance_df: Pandas DataFrame with distances between all tower pairs
    """
    # Get list of call signs
    call_signs = list(towers_dict.keys())
    num_towers = len(call_signs)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((num_towers, num_towers))
    
    # Calculate distances for all pairs
    for i, call_sign1 in enumerate(call_signs):
        lat1 = towers_dict[call_sign1][0]
        lon1 = towers_dict[call_sign1][1]
        
        for j, call_sign2 in enumerate(call_signs):
            # Skip if same tower or already calculated
            if i <= j:
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    lat2 = towers_dict[call_sign2][0]
                    lon2 = towers_dict[call_sign2][1]
                    distance = haversine_distance(lat1, lon1, lat2, lon2)
                    # Store distance in both positions of the matrix
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance
    
    # Convert to pandas DataFrame for better readability
    distance_df = pd.DataFrame(distance_matrix, index=call_signs, columns=call_signs)
    return distance_df

def create_adjacency_matrix(distance_df, towers_dict):
    """
    Convert distance matrix to binary adjacency matrix:
    - 1 if distance <= threshold
    - 0 if distance > threshold
    
    Reads individual thresholds for each tower from a CSV file.
    """
    # Read thresholds from CSV file
    thresholds = {}
    call_signs = distance_df.index.tolist()
    
    # Create a copy of the distance matrix
    adjacency_df = distance_df.copy()
    
    # Apply tower-specific thresholds to create binary adjacency matrix
    for i, tower in enumerate(adjacency_df.index):
        threshold_km = towers_dict.get(tower)[2]
        # Apply threshold for this specific tower's connections
        adjacency_df.iloc[i, :] = (adjacency_df.iloc[i, :] <= threshold_km).astype(int)
    
    # Set diagonal to 0 (no self-loops)
    for i in range(len(adjacency_df)):
        adjacency_df.iloc[i, i] = 0
        
    return adjacency_df
        
        
def create_network_graph(adjacency_df, towers_dict=None, threshold_csv_path=None):
    """
    Create a network graph from the adjacency matrix with greedy coloring
    
    Args:
        adjacency_df: Binary adjacency matrix
        towers_dict: Dictionary with call signs as keys and (lat, long) tuples as values
        threshold_csv_path: Path to CSV file containing threshold values
    """
    # Create a graph
    G = nx.from_pandas_adjacency(adjacency_df)
    
    # Set node positions based on geographic coordinates if available
    if towers_dict:
        pos = {}
        for node in G.nodes():
            if node in towers_dict:
                # Use longitude for x and latitude for y
                lon, lat = towers_dict[node][1], towers_dict[node][0]
                pos[node] = (lon, lat)
        nx.set_node_attributes(G, pos, 'pos')
    else:
        # Use spring layout if no coordinates provided
        pos = nx.spring_layout(G)
        nx.set_node_attributes(G, pos, 'pos')
    
    # Apply greedy coloring algorithm
    colors_dict = nx.coloring.greedy_color(G, strategy="largest_first")
    
    # Get unique colors for visualization
    unique_colors = sorted(set(colors_dict.values()))
    num_colors = len(unique_colors)
    print(f"Number of colors used: {num_colors}")
    
    # Create a color map for visualization
    color_map = plt.cm.get_cmap('tab20', num_colors)
    
    # Map the color indices to actual colors
    node_colors = [color_map(colors_dict[node] % 20) for node in G.nodes()]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw nodes with their colors
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    # Add title with threshold information
    if threshold_csv_path:
        plt.title(f"Radio Tower Network (Using tower-specific thresholds, Colors: {num_colors})")
    else:
        plt.title(f"Radio Tower Network (Colors: {num_colors})")
    
    # Add a legend for colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=color_map(i % 20), markersize=10) 
                     for i in range(min(num_colors, 10))]
    
    legend_labels = [f"Color {i+1}" for i in range(min(num_colors, 10))]
    if num_colors > 10:
        legend_labels[-1] = f"Color 10 (+ {num_colors - 10} more)"
    
    plt.legend(legend_handles, legend_labels, loc='lower right', title="Channel Groups")
    
    # Remove axis
    plt.axis('off')
    
    # Save and show
    plt.tight_layout()
    plt.savefig("radio_tower_network_colored.png", dpi=300)
    plt.show()
    
    return G, colors_dict

def analyze_coloring(G, colors_dict):
    """
    Analyze the coloring of the graph
    """
    print("\nColoring Analysis:")
    
    # Count the number of nodes with each color
    color_counts = {}
    for node, color in colors_dict.items():
        if color not in color_counts:
            color_counts[color] = []
        color_counts[color].append(node)
    
    # Print information about each color
    print(f"Total colors used: {len(color_counts)}")
    
    for color, nodes in sorted(color_counts.items()):
        print(f"Color {color}: {len(nodes)} stations - {', '.join(nodes[:3])}" + 
              (f" and {len(nodes)-3} more" if len(nodes) > 3 else ""))
    
    # Check if the coloring is valid
    valid = True
    for u, v in G.edges():
        if colors_dict[u] == colors_dict[v]:
            valid = False
            print(f"Invalid coloring: Adjacent nodes {u} and {v} have the same color {colors_dict[u]}")
    
    if valid:
        print("The coloring is valid - no adjacent nodes have the same color.")
    else:
        print("WARNING: The coloring is invalid - some adjacent nodes have the same color.")

def analyze_network(G):
    """
    Analyze the network graph and print some statistics
    """
    print("\nNetwork Analysis:")
    print(f"Number of nodes (towers): {G.number_of_nodes()}")
    print(f"Number of edges (connections): {G.number_of_edges()}")
    
    # Calculate network density
    density = nx.density(G)
    print(f"Network density: {density:.4f}")
    
    # Find connected components
    components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(components)}")
    
    # Find largest component
    largest_component = max(components, key=len)
    print(f"Size of largest component: {len(largest_component)} towers")
    
    # Calculate average degree
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    print(f"Average degree (connections per tower): {avg_degree:.2f}")
    
    # Find towers with highest degree (most connections)
    degree_dict = dict(G.degree())
    sorted_degrees = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 towers by number of connections:")
    for i, (node, degree) in enumerate(sorted_degrees[:5]):
        print(f"{i+1}. {node}: {degree} connections")

def load_tower_data(file_path):
    """
    Load tower data from a CSV file
    Expected format: call_sign,latitude,longitude
    """
    towers_dict = {}
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header row if present
        
        for row in reader:
            if len(row) >= 3:
                call_sign = row[0].strip()
                try:
                    lat = float(row[1].replace("\xa0", "").strip())
                    lon = float(row[2].replace("\xa0", "").strip())
                    radius = float(row[3].replace("\xa0", "").strip()) if len(row) > 3 else 13.7  # Default radius
                    towers_dict[call_sign] = (lat, lon, radius)
                except ValueError:
                    print(f"Warning: Invalid coordinates for {call_sign}, skipping.")
    
    return towers_dict

def main():
    print("Radio Tower Network Graph Generator with Greedy Coloring")
    print("-------------------------------------------------------")
    
    # Threshold in miles
    threshold_miles = 13.7
    
    # Example dictionary of radio towers
    # Format: 'CALL_SIGN': (latitude, longitude)
    example_towers = {
        'KLKX-LP': (45.8853, -95.3774),  # Alexandria, MN
        'KSWJ': (45.8888, -95.3679),      # Alexandria, MN
        'KULO': (45.8724, -95.3452),      # Alexandria, MN
        'KCLD-FM': (45.5579, -94.1632),   # St. Cloud, MN
        'KNSI': (45.5469, -94.2023),      # St. Cloud, MN
        'KVSC': (45.5501, -94.1494),      # St. Cloud, MN
        'WJON': (45.5576, -94.1708),      # St. Cloud, MN
        'KZRV': (45.6201, -94.2024),      # Sartell, MN
        'WBHR': (45.5912, -94.1656),      # Sauk Rapids, MN
        'WHMH-FM': (45.5934, -94.1701),   # Sauk Rapids, MN
        'KLZZ': (45.5345, -94.2344),      # Waite Park, MN
        'KCML': (45.5631, -94.3190),      # St. Joseph, MN
    }
    
    print("\nOptions:")
    print("1. Use example tower data")
    print("2. Load tower data from CSV file")
    print("3. Enter tower data manually")
    
    choice = input("\nEnter your choice (1-3): ")
    
    towers_dict = {}
    
    if choice == '1':
        towers_dict = example_towers
        print(f"Using example data with {len(towers_dict)} towers.")
    
    elif choice == '2':
        file_path = input("Enter the CSV file path: ")
        towers_dict = load_tower_data(file_path)
        print(f"Loaded {len(towers_dict)} towers from {file_path}.")
    
    elif choice == '3':
        num_towers = int(input("Enter number of towers to input: "))
        
        for i in range(num_towers):
            print(f"\nTower {i+1}:")
            call_sign = input("Call Sign: ")
            lat = float(input("Latitude: "))
            lon = float(input("Longitude: "))
            towers_dict[call_sign] = (lat, lon)
    
    else:
        print("Invalid choice. Using example data.")
        towers_dict = example_towers
    
    # Calculate distance matrix
    print("\nCalculating distances between all towers...")
    distance_df = calculate_distance_matrix(towers_dict)
    
    print("\nDistance Matrix (kilometers, sample):")
    print(distance_df.iloc[:5, :5].round(2))  # Show just a sample
    
    # Convert to adjacency matrix based on threshold
    adjacency_df = create_adjacency_matrix(distance_df, towers_dict)
    
    print(f"\nAdjacency Matrix (1 = within {threshold_miles} miles, 0 = beyond, sample):")
    print(adjacency_df.iloc[:5, :5])  # Show just a sample
    
    # Create and visualize network graph with coloring
    print("\nCreating network graph with greedy coloring...")
    G, colors_dict = create_network_graph(adjacency_df, towers_dict, "US_data_With_freq.csv")
    
    # Network analysis
    analyze_network(G)
    
    # Coloring analysis
    analyze_coloring(G, colors_dict)
    
    # Save adjacency matrix and coloring results
    save_choice = input("\nSave results to CSV? (y/n): ")
    if save_choice.lower() == 'y':
        # Save adjacency matrix
        matrix_file = input("Enter adjacency matrix filename (default: tower_adjacency.csv): ") or "tower_adjacency.csv"
        adjacency_df.to_csv(matrix_file)
        print(f"Adjacency matrix saved to {matrix_file}")
        
        # Save coloring results
        colors_file = input("Enter coloring results filename (default: tower_colors.csv): ") or "tower_colors.csv"
        colors_df = pd.DataFrame(list(colors_dict.items()), columns=['Call_Sign', 'Color_Group'])
        colors_df.sort_values(by=['Color_Group', 'Call_Sign'], inplace=True)
        colors_df.to_csv(colors_file, index=False)
        print(f"Coloring results saved to {colors_file}")

if __name__ == "__main__":
    main()

    
def createRandomPlot(NUM_NODES, RADIUS):
    # Generate random positions for the nodes
    nodes = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_NODES)]

    # Function to calculate Euclidean distance
    def euclidean_distance(node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    # Create a graph
    graph = nx.Graph()

    # Add nodes to the graph
    for i in range(NUM_NODES):
        graph.add_node(i, pos=nodes[i])

    # Add edges if nodes are within the radius
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            if euclidean_distance(nodes[i], nodes[j]) <= RADIUS:
                graph.add_edge(i, j)

    # Graph coloring using greedy algorithm
    colors = nx.coloring.greedy_color(graph, strategy="largest_first")

    # Visualization
    pos = nx.get_node_attributes(graph, 'pos')
    node_colors = [colors[node] for node in graph.nodes]

    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.tab20, node_size=500)
    plt.title("Graph with 38 Nodes and Radius-Based Edges")
    plt.show()
#createRandomPlot(38,  13.7)