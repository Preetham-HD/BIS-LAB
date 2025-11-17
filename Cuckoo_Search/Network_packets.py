import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# --- 1. Graph Setup ---
def create_complex_graph():
    """
    Creates a complex, directed graph representing a network.
    Edges have 'cost' attributes for optimization.
    """
    G = nx.DiGraph()

    # Nodes (e.g., routers)
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    G.add_nodes_from(nodes)

    # Edges with costs (e.g., latency in ms)
    # (source, target, cost)
    edges = [
        ('A', 'B', 10), ('A', 'C', 15), ('A', 'D', 25),
        ('B', 'E', 20), ('B', 'F', 5),
        ('C', 'E', 10), ('C', 'G', 40),
        ('D', 'F', 30), ('D', 'H', 15),
        ('E', 'I', 10), ('E', 'G', 5),
        ('F', 'I', 15), ('F', 'H', 20),
        ('G', 'J', 10),
        ('H', 'J', 12),
        ('I', 'J', 8),
        # Backwards/Alternative paths for complexity
        ('B', 'A', 12), ('C', 'B', 5), ('E', 'C', 15),
        ('H', 'E', 25), ('I', 'G', 15)
    ]

    for u, v, cost in edges:
        G.add_edge(u, v, cost=cost)

    return G

# --- 2. Path Encoding and Fitness ---

def calculate_path_cost(G, path):
    """
    Calculates the total cost (fitness) of a path.
    A path is a list of nodes, e.g., ['A', 'B', 'E', 'I', 'J'].
    Returns a very high cost for invalid paths.
    """
    if not path or len(path) < 2:
        return float('inf')

    total_cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if G.has_edge(u, v):
            total_cost += G[u][v]['cost']
        else:
            # High cost for an invalid hop (broken link)
            return float('inf')

    return total_cost

def generate_random_valid_path(G, source, destination, max_length=10):
    """
    Generates a random, valid path between source and destination.
    Uses a simple random walk until the destination is reached or max_length is exceeded.
    """
    path = [source]
    current_node = source

    while current_node != destination and len(path) < max_length:
        neighbors = list(G.successors(current_node))
        if not neighbors:
            return [] # Dead end

        # Choose a random neighbor that is not the previous node (to avoid immediate cycles)
        next_node = random.choice(neighbors)
        if len(path) > 1 and next_node == path[-2]:
             # Try another one, or just take the step if only one option
             pass

        path.append(next_node)
        current_node = next_node

    return path if current_node == destination else []


def levy_flight(path, G, destination):
    """
    Simulates a 'Lévy Flight' for path optimization.
    This creates a new "egg" (path) from an existing one, promoting global search.

    The discrete adaptation works by:
    1. Randomly selecting a segment of the current path (local search/mutation).
    2. Generating a new random path segment (Lévy jump) from the selected node
       to a point near the destination, then continuing to the destination.
    """
    if len(path) < 2: return generate_random_valid_path(G, path[0], destination)

    # 1. Select a random crossover point (excluding source and destination)
    if len(path) > 2:
        crossover_idx = random.randint(1, len(path) - 2)
    else:
        crossover_idx = 1 # Only one hop

    start_node = path[crossover_idx]

    # 2. Randomly decide to jump to a random neighbor or make a long jump
    if random.random() < 0.7: # Small step (Local search)
        new_segment = generate_random_valid_path(G, start_node, destination, max_length=5)
    else: # Large step (Global search, via a randomly chosen intermediate node)
        intermediate_nodes = [n for n in G.nodes() if n not in [start_node, destination]]
        if intermediate_nodes:
            intermediate_node = random.choice(intermediate_nodes)

            # Segment 1: Start node to intermediate node
            try:
                seg1 = nx.shortest_path(G, start_node, intermediate_node, weight='cost')
            except nx.NetworkXNoPath:
                seg1 = [start_node] # Fallback if no path

            # Segment 2: Intermediate node to destination
            try:
                seg2 = nx.shortest_path(G, intermediate_node, destination, weight='cost')
            except nx.NetworkXNoPath:
                 seg2 = [destination]

            # Combine, removing duplicates
            new_segment = seg1 + seg2[1:]
        else:
             new_segment = generate_random_valid_path(G, start_node, destination, max_length=5)

    if not new_segment or new_segment[0] != start_node or new_segment[-1] != destination:
         # If new segment failed, try a totally new path from the start node of the segment
         new_segment = generate_random_valid_path(G, start_node, destination)

    # 3. Create the new solution: old path up to crossover, plus the new segment
    new_path = path[:crossover_idx] + new_segment

    # 4. Cleanup: Remove immediate cycles (e.g., A -> B -> A -> C)
    cleaned_path = []
    for node in new_path:
        if cleaned_path and node == cleaned_path[-1]: continue # Skip duplicates
        if len(cleaned_path) >= 2 and node == cleaned_path[-2]:
            cleaned_path.pop() # Remove the previous node to clear the cycle

        cleaned_path.append(node)

    return cleaned_path if cleaned_path[-1] == destination else []

# --- 3. Cuckoo Search Algorithm ---

def cuckoo_search_path_optimization(G, source, destination, N_nests=10, Pa=0.25, max_iter=100):

    print(f"\n--- Starting Cuckoo Search: {source} to {destination} ---\n")
    print(f"Parameters: Nests={N_nests}, Pa={Pa}, Max Iter={max_iter}\n")

    # Initialization: Generate an initial population of host nests (paths)
    nests = []
    for _ in range(N_nests):
        path = generate_random_valid_path(G, source, destination)
        if path:
            cost = calculate_path_cost(G, path)
            nests.append({'path': path, 'cost': cost})
        else:
            nests.append({'path': [], 'cost': float('inf')})

    # Sort nests by cost (best is lowest cost)
    nests.sort(key=lambda x: x['cost'])

    best_path = nests[0]['path']
    best_cost = nests[0]['cost']

    if best_cost == float('inf'):
        print("Error: Could not find any initial valid path. Adjust graph or path generation.")
        return best_path, best_cost

    print(f"Initial Best Cost: {best_cost}, Path: {' -> '.join(best_path)}")
    print("-" * 50)

    # Main optimization loop
    for t in range(max_iter):
        # 1. Generate a new cuckoo egg (solution) via Lévy flight
        idx_cuckoo = random.randint(0, N_nests - 1)
        current_nest_path = nests[idx_cuckoo]['path']

        # Ensure we have a path to start the flight
        if not current_nest_path or current_nest_path[-1] != destination:
             current_nest_path = generate_random_valid_path(G, source, destination)
             if not current_nest_path: continue # Skip if path can't be generated

        new_cuckoo_path = levy_flight(current_nest_path, G, destination)
        new_cuckoo_cost = calculate_path_cost(G, new_cuckoo_path)

        # 2. Choose a random nest (j) to compare with
        idx_host = random.randint(0, N_nests - 1)
        host_cost = nests[idx_host]['cost']

        # 3. Replace nest (j) if the new cuckoo egg is better
        if new_cuckoo_cost < host_cost:
            nests[idx_host] = {'path': new_cuckoo_path, 'cost': new_cuckoo_cost}

        # 4. Abandon a fraction (Pa) of worse nests and build new ones
        num_abandon = int(N_nests * Pa)
        # The nests are already sorted, so we abandon the last 'num_abandon' ones (worst solutions)

        for i in range(1, num_abandon + 1):
            idx_worst = N_nests - i
            new_path = generate_random_valid_path(G, source, destination)
            if new_path:
                 new_cost = calculate_path_cost(G, new_path)
                 nests[idx_worst] = {'path': new_path, 'cost': new_cost}
            else:
                 # In case new path generation fails, keep the original worst solution
                 pass

        # 5. Re-sort the nests and update the global best
        nests.sort(key=lambda x: x['cost'])

        current_best_cost = nests[0]['cost']
        current_best_path = nests[0]['path']

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_path = current_best_path

        # Print iteration results
        print(f"Iter {t+1:03d}: Current Best Cost: {current_best_cost}, Path: {' -> '.join(current_best_path)}")


    print("-" * 50)
    return best_path, best_cost


# --- 4. Main Execution and Visualization ---

# 1. Create the complex graph
NETWORK_GRAPH = create_complex_graph()
SOURCE_NODE = 'A'
DESTINATION_NODE = 'J'

print("--- Network Graph Definition ---")

# 2. Print the graph details
print("Nodes:", list(NETWORK_GRAPH.nodes()))
print("Edges and Costs (Sample):")
for u, v, data in list(NETWORK_GRAPH.edges(data=True))[:5]:
    print(f"  {u} -> {v}: Cost {data['cost']}")
print(f"Total Edges: {NETWORK_GRAPH.number_of_edges()}")

# 3. Visualize the graph
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(NETWORK_GRAPH, seed=42) # Layout for visualization
nx.draw(NETWORK_GRAPH, pos, with_labels=True, node_size=1500, node_color='lightblue',
        font_size=10, font_weight='bold', arrowsize=20)

# Draw edge labels (costs)
edge_labels = nx.get_edge_attributes(NETWORK_GRAPH, 'cost')
nx.draw_networkx_edge_labels(NETWORK_GRAPH, pos, edge_labels=edge_labels, font_color='red')

# Highlight source and destination
nx.draw_networkx_nodes(NETWORK_GRAPH, pos, nodelist=[SOURCE_NODE], node_color='green', node_size=2000, label='Source')
nx.draw_networkx_nodes(NETWORK_GRAPH, pos, nodelist=[DESTINATION_NODE], node_color='red', node_size=2000, label='Destination')

plt.title("Complex Network Graph for Packet Optimization")
#
plt.show()

# 4. Run the Cuckoo Search optimization
FINAL_BEST_PATH, FINAL_BEST_COST = cuckoo_search_path_optimization(
    G=NETWORK_GRAPH,
    source=SOURCE_NODE,
    destination=DESTINATION_NODE,
    N_nests=20, # Number of solutions to maintain
    Pa=0.25,    # Probability of abandoning the worst nest
    max_iter=50  # Number of generations
)

# 5. Print the final best result
print("\n" * 2)
print("=" * 60)
print(f"FINAL BEST PATH FOUND BY CUCKOO SEARCH")
print(f"   BEST PATH: {' -> '.join(FINAL_BEST_PATH)}")
print(f"   BEST COST: {FINAL_BEST_COST}")
print("=" * 60)



# Output:
#  --- Network Graph Definition ---
# Nodes: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
# Edges and Costs (Sample):
#   A -> B: Cost 10
#   A -> C: Cost 15
#   A -> D: Cost 25
#   B -> E: Cost 20
#   B -> F: Cost 5
# Total Edges: 21

# --- Starting Cuckoo Search: A to J ---

# Parameters: Nests=20, Pa=0.25, Max Iter=50

# Initial Best Cost: 38, Path: A -> B -> F -> I -> J
# --------------------------------------------------
# Iter 001: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 002: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 003: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 004: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 005: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 006: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 007: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 008: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 009: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 010: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 011: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 012: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 013: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 014: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 015: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 016: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 017: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 018: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 019: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 020: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 021: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 022: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 023: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 024: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 025: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 026: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 027: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 028: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 029: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 030: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 031: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 032: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 033: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 034: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 035: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 036: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 037: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 038: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 039: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 040: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 041: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 042: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 043: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 044: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 045: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 046: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 047: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 048: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 049: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# Iter 050: Current Best Cost: 38, Path: A -> B -> F -> I -> J
# --------------------------------------------------



# ============================================================
# FINAL BEST PATH FOUND BY CUCKOO SEARCH
#    BEST PATH: A -> B -> F -> I -> J
#    BEST COST: 38
# ============================================================
