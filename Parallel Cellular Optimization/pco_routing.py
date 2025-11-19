import networkx as nx
import random
import time
import pandas as pd
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 1. Define the network structure
G = nx.DiGraph()
edges = [
    ('A', 'B', {'cost': 1, 'capacity': 10}),
    ('A', 'C', {'cost': 3, 'capacity': 15}),
    ('B', 'C', {'cost': 1, 'capacity': 10}),
    ('B', 'D', {'cost': 2, 'capacity': 10}),
    ('C', 'D', {'cost': 1, 'capacity': 15}),
    ('C', 'E', {'cost': 4, 'capacity': 10}),
    ('D', 'E', {'cost': 1, 'capacity': 20})
]
G.add_edges_from(edges)

# Define source-destination pairs for routing attempts
source_destination_pairs = [
    ('A', 'D'),
    ('A', 'E'),
    ('B', 'E'),
    ('C', 'D')
]

# Initialize CA grid state and simulation parameters
initial_pheromone = 0.1
ca_grid_state_initial = {}
for u, v in G.edges():
    ca_grid_state_initial[(u, v)] = {
        'is_active_route': 0,
        'pheromone_level': initial_pheromone,
        'traffic_load': 0
    }

simulation_parameters = {
    'num_time_steps': 100,
    'pheromone_evaporation_rate': 0.05,
    'pheromone_deposit_amount': 0.1,
    'random_exploration_factor': 0.01,
    'traffic_increase_factor': 0.01,
    'traffic_decrease_factor': 0.02,
    'route_deactivation_threshold': 0.8,
    'pheromone_deposit_threshold': 0.2,
    'activation_factor': 0.05,
    'routing_attempts_per_step': 5,
    'pheromone_deposit_on_success': 0.5,
    'dijkstra_traffic_factor': 0.5,
    'dijkstra_pheromone_factor': 0.05
}

# Helper function to get neighborhood (from previous step)
def get_neighborhood(u, v, graph):
    neighborhood = []
    for in_edge_u, in_edge_v in graph.in_edges(u):
        neighborhood.append((in_edge_u, in_edge_v))
    for out_edge_u, out_edge_v in graph.out_edges(v):
        neighborhood.append((out_edge_u, out_edge_v))
    return neighborhood

# Helper function to update cell state (from previous step)
def update_cell_state(u, v, current_state, neighbor_states, graph, simulation_parameters):
    new_state = current_state.copy()
    pheromone_evaporation_rate = simulation_parameters['pheromone_evaporation_rate']
    pheromone_deposit_amount = simulation_parameters['pheromone_deposit_amount']
    random_exploration_factor = simulation_parameters['random_exploration_factor']
    traffic_increase_factor = simulation_parameters['traffic_increase_factor']
    traffic_decrease_factor = simulation_parameters['traffic_decrease_factor']
    route_deactivation_threshold = simulation_parameters['route_deactivation_threshold']

    new_state['pheromone_level'] *= (1 - pheromone_evaporation_rate)

    for (nu, nv), state in neighbor_states.items():
        if state['pheromone_level'] > simulation_parameters['pheromone_deposit_threshold']:
            new_state['pheromone_level'] += pheromone_deposit_amount * state['pheromone_level']

    max_pheromone = simulation_parameters.get('max_pheromone', float('inf'))
    new_state['pheromone_level'] = min(new_state['pheromone_level'], max_pheromone)

    active_neighbors = sum(state['is_active_route'] for state in neighbor_states.values())
    new_state['traffic_load'] += active_neighbors * traffic_increase_factor
    new_state['traffic_load'] *= (1 - traffic_decrease_factor)

    edge_capacity = graph[u][v].get('capacity', float('inf'))
    new_state['traffic_load'] = min(new_state['traffic_load'], edge_capacity)
    graph[u][v]['traffic_load'] = new_state['traffic_load']

    cost = graph[u][v].get('cost', 1)
    available_capacity = edge_capacity - new_state['traffic_load']
    attractiveness = (new_state['pheromone_level'] + 1e-6) / (cost + 1e-6) * (available_capacity + 1e-6)

    activation_probability = attractiveness * simulation_parameters['activation_factor']

    if random.random() < random_exploration_factor:
        activation_probability = random.random()

    if new_state['is_active_route'] == 0:
        if random.random() < activation_probability:
            new_state['is_active_route'] = 1
    else:
        if new_state['traffic_load'] > edge_capacity * route_deactivation_threshold:
            new_state['is_active_route'] = 0
        elif random.random() > activation_probability and random.random() > (1 - random_exploration_factor):
            new_state['is_active_route'] = 0

    return new_state

# Pathfinding function using Dijkstra's algorithm (optimized version)
def find_route_with_dijkstra_ca(source, destination, graph, ca_grid_state):
    def get_dynamic_weight(u, v):
        edge_state = ca_grid_state.get((u, v), {'is_active_route': 0, 'pheromone_level': 0, 'traffic_load': float('inf')})
        original_cost = graph[u][v].get('cost', 1)
        traffic_load = edge_state.get('traffic_load', 0)
        pheromone_level = edge_state.get('pheromone_level', 0)

        alpha = simulation_parameters.get('dijkstra_traffic_factor', 0.1)
        beta = simulation_parameters.get('dijkstra_pheromone_factor', 0.01)

        dynamic_weight = original_cost + alpha * traffic_load - beta * pheromone_level
        return max(0, dynamic_weight)

    distances = {node: float('inf') for node in graph.nodes()}
    distances[source] = 0
    previous_nodes = {}
    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == destination:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes.get(current_node)
            return path[::-1]

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge = (current_node, neighbor)
            if edge in graph.edges():
                weight = get_dynamic_weight(current_node, neighbor)
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

    return None

# Parallel update function (from previous step)
def update_cell_state_parallel(edge_chunk, ca_grid_state, graph, simulation_parameters):
    updated_chunk_state = {}
    for u, v in edge_chunk:
        current_state = ca_grid_state[(u, v)]
        neighborhood = get_neighborhood(u, v, graph)
        neighbor_states = {
            (nu, nv): ca_grid_state.get((nu, nv), {'is_active_route': 0, 'pheromone_level': 0, 'traffic_load': 0})
            for (nu, nv) in neighborhood
        }
        new_state = update_cell_state(u, v, current_state, neighbor_states, graph, simulation_parameters)
        updated_chunk_state[(u, v)] = new_state
    return updated_chunk_state

# Metrics calculation function (from previous step)
def calculate_metrics(ca_grid_history, graph, simulation_parameters, successful_routes_history):
    num_time_steps = simulation_parameters['num_time_steps']
    metrics = {
        'time_step': list(range(num_time_steps + 1)),
        'num_successful_routes': [0],
        'avg_route_cost': [0],
        'avg_path_length': [0],
        'avg_traffic_load': [],
        'num_overloaded_edges': []
    }

    for t in range(num_time_steps + 1):
        current_ca_state = ca_grid_history[t]

        total_traffic_load = 0
        overloaded_edges_count = 0
        num_edges = 0
        for (u, v), state in current_ca_state.items():
            num_edges += 1
            traffic_load = state.get('traffic_load', 0)
            capacity = graph[u][v].get('capacity', float('inf'))

            total_traffic_load += traffic_load
            if capacity != float('inf') and traffic_load > capacity:
                overloaded_edges_count += 1

        metrics['avg_traffic_load'].append(total_traffic_load / num_edges if num_edges > 0 else 0)
        metrics['num_overloaded_edges'].append(overloaded_edges_count)

        if t > 0:
            successful_routes_this_step = successful_routes_history[t-1]
            metrics['num_successful_routes'].append(len(successful_routes_this_step))

            if successful_routes_this_step:
                total_cost_this_step = 0
                total_length_this_step = 0
                for source, destination, route in successful_routes_this_step:
                    route_cost = sum(graph[route[i]][route[i+1]].get('cost', 1) for i in range(len(route) - 1))
                    total_cost_this_step += route_cost
                    total_length_this_step += len(route) - 1

                metrics['avg_route_cost'].append(total_cost_this_step / len(successful_routes_this_step))
                metrics['avg_path_length'].append(total_length_this_step / len(successful_routes_this_step))
            else:
                metrics['avg_route_cost'].append(0)
                metrics['avg_path_length'].append(0)

    return metrics

# Main simulation runner (adapted from previous step)
def run_simulation_optimized_routing(graph, initial_ca_grid_state, simulation_parameters, source_destination_pairs, parallel=True):
    if parallel:
        print("\nStarting Optimized Parallel Cellular Automaton Simulation with Routing...")
    else:
        print("\nStarting Optimized Sequential Cellular Automaton Simulation with Routing...")

    ca_grid_state_run = initial_ca_grid_state.copy()
    ca_grid_history_run = [ca_grid_state_run.copy()]
    successful_routes_history_run = []

    start_time = time.time()

    for t in range(simulation_parameters['num_time_steps']):
        for edge in graph.edges():
            ca_grid_state_run[edge]['is_active_route'] = 0

        successful_routes_this_step = []
        for _ in range(simulation_parameters['routing_attempts_per_step']):
            source, destination = random.choice(source_destination_pairs)
            found_route = find_route_with_dijkstra_ca(source, destination, graph, ca_grid_state_run)

            if found_route:
                successful_routes_this_step.append((source, destination, found_route))
                for i in range(len(found_route) - 1):
                    u, v = found_route[i], found_route[i+1]
                    edge = (u, v)
                    if edge in ca_grid_state_run:
                        ca_grid_state_run[edge]['pheromone_level'] += simulation_parameters['pheromone_deposit_on_success']
                        ca_grid_state_run[edge]['is_active_route'] = 1
                        ca_grid_state_run[edge]['traffic_load'] += 1
                        edge_capacity = graph[u][v].get('capacity', float('inf'))
                        ca_grid_state_run[edge]['traffic_load'] = min(ca_grid_state_run[edge]['traffic_load'], edge_capacity)
                        graph[u][v]['traffic_load'] = ca_grid_state_run[edge]['traffic_load']

        successful_routes_history_run.append(successful_routes_this_step)

        next_ca_grid_state_run = {}
        edges_list = list(graph.edges())

        if parallel:
            num_processes = multiprocessing.cpu_count()
            chunk_size = max(1, len(edges_list) // num_processes)
            edge_chunks = [edges_list[i:i + chunk_size] for i in range(0, len(edges_list), chunk_size)]

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                future_to_chunk = {
                    executor.submit(update_cell_state_parallel, chunk, ca_grid_state_run.copy(), graph, simulation_parameters): chunk
                    for chunk in edge_chunks
                }

                for future in as_completed(future_to_chunk):
                    try:
                        updated_chunk_state = future.result()
                        next_ca_grid_state_run.update(updated_chunk_state)
                    except Exception as exc:
                        print(f'Edge chunk generated an exception: {exc}')
        else:
            for u, v in graph.edges():
                current_state = ca_grid_state_run[(u, v)]
                neighborhood = get_neighborhood(u, v, graph)
                neighbor_states = {
                    (nu, nv): ca_grid_state_run.get((nu, nv), {'is_active_route': 0, 'pheromone_level': 0, 'traffic_load': 0})
                    for (nu, nv) in neighborhood
                }
                new_state = update_cell_state(u, v, current_state, neighbor_states, graph, simulation_parameters)
                next_ca_grid_state_run[(u, v)] = new_state

        ca_grid_state_run = next_ca_grid_state_run
        ca_grid_history_run.append(ca_grid_state_run.copy())

        if (t + 1) % 10 == 0:
            if parallel:
                print(f"Optimized Parallel Time Step {t + 1}/{simulation_parameters['num_time_steps']} completed.")
            else:
                print(f"Optimized Sequential Time Step {t + 1}/{simulation_parameters['num_time_steps']} completed.")

    end_time = time.time()
    execution_time = end_time - start_time
    if parallel:
        print(f"Optimized Parallel Simulation Finished in {execution_time:.4f} seconds.")
    else:
        print(f"Optimized Sequential Simulation Finished in {execution_time:.4f} seconds.")

    return ca_grid_history_run, successful_routes_history_run, execution_time

# --- Run optimized simulations and compare ---
# Reset graph traffic load before running simulations
for u,v in G.edges():
    G[u][v]['traffic_load'] = 0

# Run optimized parallel simulation
ca_grid_history_opt_parallel, successful_routes_history_opt_parallel, execution_time_opt_parallel = run_simulation_optimized_routing(
    G.copy(),
    ca_grid_state_initial.copy(),
    simulation_parameters,
    source_destination_pairs,
    parallel=True
)

# Reset graph traffic load for sequential run
for u,v in G.edges():
    G[u][v]['traffic_load'] = 0

# Run optimized sequential simulation
ca_grid_history_opt_sequential, successful_routes_history_opt_sequential, execution_time_opt_sequential = run_simulation_optimized_routing(
    G.copy(),
    ca_grid_state_initial.copy(),
    simulation_parameters,
    source_destination_pairs,
    parallel=False
)

# Calculate metrics for optimized simulations
print("\nCalculating Metrics for Optimized Simulations...")
parallel_metrics_opt = calculate_metrics(ca_grid_history_opt_parallel, G, simulation_parameters, successful_routes_history_opt_parallel)
sequential_metrics_opt = calculate_metrics(ca_grid_history_opt_sequential, G, simulation_parameters, successful_routes_history_opt_sequential)

parallel_metrics_opt_df = pd.DataFrame(parallel_metrics_opt)
sequential_metrics_opt_df = pd.DataFrame(sequential_metrics_opt)

print("\nOptimized Parallel Metrics:")
display(parallel_metrics_opt_df.head())

print("\nOptimized Sequential Metrics:")
display(sequential_metrics_opt_df.head())

print(f"\nOptimized Parallel Execution Time: {execution_time_opt_parallel:.4f} seconds")
print(f"Optimized Sequential Execution Time: {execution_time_opt_sequential:.4f} seconds")

# Compare average successful routes
avg_successful_routes_opt_parallel = parallel_metrics_opt_df['num_successful_routes'].mean()
avg_successful_routes_opt_sequential = sequential_metrics_opt_df['num_successful_routes'].mean()

print(f"\nAverage Successful Routes per Step (Optimized Parallel): {avg_successful_routes_opt_parallel:.2f}")
print(f"Average Successful Routes per Step (Optimized Sequential): {avg_successful_routes_opt_sequential:.2f}")

# Compare average traffic load at the end
avg_traffic_end_opt_parallel = parallel_metrics_opt_df['avg_traffic_load'].iloc[-1]
avg_traffic_end_opt_sequential = sequential_metrics_opt_df['avg_traffic_load'].iloc[-1]

print(f"\nAverage Traffic Load at End (Optimized Parallel): {avg_traffic_end_opt_parallel:.2f}")
print(f"Average Traffic Load at End (Optimized Sequential): {avg_traffic_end_opt_sequential:.2f}")
