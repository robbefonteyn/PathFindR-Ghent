import os
import osmnx as ox
import networkx as nx
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from ipywidgets import interact, IntSlider
from itertools import permutations
from random import choice, random
import random

def translate_address(graph, geolocator, address: str, return_coords=False):
    """
    Translate an address to a corresponding node in the specified graph.
    """

    # Geocode the address to get latitude and longitude
    location = geolocator.geocode(address)
    if location is None:
        raise ValueError(f"Address '{address}' could not be geocoded.")

    start_lat, start_lon = location.latitude, location.longitude

    # Find the nearest node in the graph
    node = ox.distance.nearest_nodes(graph, X=start_lon, Y=start_lat)
    if node is None:
        raise ValueError("No nearest node found in the graph for the given address.")

    if return_coords:
        return node, (start_lon, start_lat)
    else:
        return node
    


def plot_path_GA(graph, path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the base graph
    ox.plot_graph(graph, ax=ax, show=False, close=False, bgcolor='white', edge_color='black', node_color='white', node_size=1, edge_linewidth=0.5)

    # Gather all edges for the path
    edges = []
    for i in range(len(path) - 1):
        route = nx.shortest_path(graph, path[i], path[i + 1], weight='length')
        edges += list(zip(route[:-1], route[1:]))
    
    # Get node coordinates for plotting
    node_positions = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes}
    edge_positions = [(node_positions[edge[0]], node_positions[edge[1]]) for edge in edges]

    # Plot the path edges
    for start, end in edge_positions:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2, zorder=3)

    # Highlight the nodes in the path
    path_coords = [node_positions[node] for node in path]
    path_x, path_y = zip(*path_coords)
    ax.scatter(path_x, path_y, color='blue', s=50, zorder=4)

    # Annotate nodes
    for idx, (x, y) in enumerate(path_coords):
        ax.text(x, y, str(idx), fontsize=6, ha='center', va='center', color='black', zorder=5)

    ax.set_title("Path Visualization on Graph")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def visualize_GA(graph, history, best_lengths, target_distance):
    @interact(generation=IntSlider(min=0, max=len(history)-1, step=1, value=len(history)-1))
    def show_generation(generation):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot the graph and path for the selected generation
        plot_path_GA(graph, history[generation], ax=ax1)
        ax1.set_title(f"Generation: {generation}, Best Length: {best_lengths[generation]:.2f}")

        # Plot the best lengths up to the current generation
        ax2.plot(range(1, len(best_lengths) + 1), best_lengths, linestyle='-', color='blue', label='Best Lengths')
        ax2.scatter(generation + 1, best_lengths[generation], color='red', s=100, zorder=3, label='Selected Generation')
        ax2.axvline(x=generation + 1, color='red', linestyle='--', linewidth=1)
        ax2.axhline(y=target_distance, color='green', linestyle='--', linewidth=2, label='Target Distance')

        # Dynamically set y-axis limits to include target_distance and best_lengths
        min_y = min(min(best_lengths), target_distance) * 0.95
        max_y = max(max(best_lengths), target_distance) * 1.05
        ax2.set_ylim(min_y, max_y)

        # Customize the plot
        ax2.set_xlim(1, len(best_lengths))
        ax2.set_title("Best Path Lengths Across Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Path Length")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()


def calculate_route_distance(graph, route):
    """Calculate the total distance of a route."""
    distance = 0
    for i in range(len(route) - 1):
        try:
            distance += nx.shortest_path_length(graph, source=route[i], target=route[i + 1], weight='length')
        except nx.NetworkXNoPath:
            distance += float('inf')  # Penalize routes with no valid path
    return distance


def build_valid_route(graph, route):
    """Ensure all consecutive nodes in the route are connected."""
    valid_route = []
    for i in range(len(route) - 1):
        try:
            # Find shortest path between two consecutive nodes
            sub_path = nx.shortest_path(graph, source=route[i], target=route[i + 1], weight='length')
            if valid_route:
                valid_route.extend(sub_path[1:])  # Avoid duplicating nodes
            else:
                valid_route.extend(sub_path)
        except nx.NetworkXNoPath:
            print(f"No path between {route[i]} and {route[i + 1]} in the graph.")
            return None  # Return None if any part of the route is invalid
    return valid_route

def validate_nodes(graph, route):
    """Check if all nodes in the route exist in the graph."""
    missing_nodes = [node for node in route if node not in graph.nodes]
    if missing_nodes:
        print(f"Missing nodes in graph: {missing_nodes}")
    else:
        print("All nodes in the route exist in the graph.")

def validate_edges(graph, route):
    """Check if edges exist between consecutive nodes in the route."""
    missing_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1) if not graph.has_edge(route[i], route[i + 1])]
    if missing_edges:
        print(f"Missing edges between consecutive nodes: {missing_edges}")
    else:
        print("All edges between consecutive nodes exist in the graph.")


def plot_path_BF(graph, path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the base graph using OSMnx
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color='gray', bgcolor='white', show=False)

    # Highlight the best path
    edges = []
    for i in range(len(path) - 1):
        try:
            route = nx.shortest_path(graph, path[i], path[i + 1], weight='length')
            edges += list(zip(route[:-1], route[1:]))
        except nx.NetworkXNoPath:
            print(f"No path between {path[i]} and {path[i+1]}. Skipping this segment.")

    # Draw the red path between nodes
    edge_x = []
    edge_y = []
    for edge in edges:
        x = [graph.nodes[edge[0]]['x'], graph.nodes[edge[1]]['x']]
        y = [graph.nodes[edge[0]]['y'], graph.nodes[edge[1]]['y']]
        edge_x.extend(x + [None])  # Add None to break line segments
        edge_y.extend(y + [None])

    ax.plot(edge_x, edge_y, color='red', linewidth=2, label='Best Path')

    # Annotate nodes
    for idx, node in enumerate(path):
        x, y = graph.nodes[node].get('x', 0), graph.nodes[node].get('y', 0)
        ax.scatter(x, y, color='blue', s=50, zorder=5)
        ax.text(x, y, str(idx), fontsize=8, color='black', ha='center', va='center', zorder=6)

    ax.set_title("Optimal Path Visualization")
    ax.legend()


# Visualization
def visualize_brute_force(graph, history, best_lengths, target_distance):
    """
    Visualize the progress of the brute force algorithm.

    Parameters:
    ----------
    graph : networkx.Graph
        The graph representing the network.
    history : list
        List of paths and their lengths for each iteration.
    best_lengths : list
        List of best lengths across iterations.
    target_distance : float
        The target distance for the path.
    """
    valid_best_lengths = [length for length in best_lengths if np.isfinite(length)]

    if not valid_best_lengths:
        print("No valid path lengths available to visualize.")
        return

    @interact(generation=IntSlider(min=0, max=len(history)-1, step=1, value=len(history)-1))
    def show_generation(generation):
        # Extract the path for the selected generation
        path, path_length = history[generation]

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot the graph and path for the selected generation
        plot_path_BF(graph, path, ax=ax1)
        ax1.set_title(f"Generation: {generation}, Path Length: {path_length:.2f}")

        # Plot the best lengths up to the current generation
        ax2.plot(range(1, len(valid_best_lengths) + 1), valid_best_lengths, linestyle='-', color='blue', label='Best Lengths')
        if generation < len(valid_best_lengths):
            ax2.scatter(generation + 1, valid_best_lengths[generation], color='red', s=100, zorder=3, label='Selected Generation')
        ax2.axvline(x=generation + 1, color='red', linestyle='--', linewidth=1)
        ax2.axhline(y=target_distance, color='green', linestyle='--', linewidth=2, label='Target Distance')

        # Dynamically set y-axis limits to include target_distance and best_lengths
        min_y = min(valid_best_lengths + [target_distance]) * 0.95
        max_y = max(valid_best_lengths + [target_distance]) * 1.05
        ax2.set_ylim(min_y, max_y)

        # Customize the plot
        ax2.set_xlim(1, len(valid_best_lengths))
        ax2.set_title("Best Path Lengths Across Generations")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Best Path Length")
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        plt.show()


def plot_routes(graph, routes, route_labels=None):
    """
    Plots multiple routes on a graph for comparison.

    Parameters:
    - graph: The NetworkX graph.
    - routes: List of routes (each route is a list of node IDs).
    - route_labels: Optional list of labels for each route.
    """
    if route_labels is None:
        route_labels = [f"Route {i+1}" for i in range(len(routes))]

    # Plot the base graph
    fig, ax = ox.plot_graph(graph, show=False, close=False, bgcolor="white", edge_color="lightgray", node_color="lightgray", node_size=1)

    # Define colors for the routes
    colors = ["red", "blue", "green"]
    styles = ["-", "-", "-"]

    # Plot each route with a different color and style
    for i, route in enumerate(routes):
        edges = []
        for j in range(len(route) - 1):
            subroute = nx.shortest_path(graph, route[j], route[j + 1], weight="length")
            edges += list(zip(subroute[:-1], subroute[1:]))

        edge_positions = [
            (graph.nodes[u]["x"], graph.nodes[u]["y"], graph.nodes[v]["x"], graph.nodes[v]["y"])
            for u, v in edges
        ]

        for x1, y1, x2, y2 in edge_positions:
            ax.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linestyle=styles[i % len(styles)], linewidth=2, label=route_labels[i] if j == 0 else "")

    # Add a legend
    #ax.legend(route_labels, loc="lower left")
    plt.title("Comparison of Routes on the same Graph")
    plt.show()


def plot_routes_separately(graph, routes, route_labels=None):
    """
    Plots each route separately on the graph.

    Parameters:
    - graph: The NetworkX graph.
    - routes: List of routes (each route is a list of node IDs).
    - route_labels: Optional list of labels for each route.
    """
    if route_labels is None:
        route_labels = [f"Route {i+1}" for i in range(len(routes))]

    # Set up subplots
    fig, axes = plt.subplots(1, len(routes), figsize=(18, 6))
    if len(routes) == 1:  # If there's only one route, axes is not iterable
        axes = [axes]

    # Colors for routes
    colors = ["red", "blue", "green"]

    for i, (route, ax) in enumerate(zip(routes, axes)):
        # Plot the base graph on each subplot
        ox.plot_graph(graph, ax=ax, show=False, close=False, bgcolor="white", edge_color="lightgray", node_color="lightgray", node_size=1)

        # Extract edges for the route
        edges = []
        for j in range(len(route) - 1):
            subroute = nx.shortest_path(graph, route[j], route[j + 1], weight="length")
            edges += list(zip(subroute[:-1], subroute[1:]))

        # Plot the edges for the route
        edge_positions = [
            (graph.nodes[u]["x"], graph.nodes[u]["y"], graph.nodes[v]["x"], graph.nodes[v]["y"])
            for u, v in edges
        ]
        for x1, y1, x2, y2 in edge_positions:
            ax.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=2)

        # Add title for each subplot
        ax.set_title(route_labels[i])

    plt.tight_layout()
    plt.show()
