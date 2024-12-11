import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from joblib import Parallel, delayed

# Constants for calorie calculation
CALORIES_PER_METER_PER_KG = 0.00008  # Average calories burned per meter walked per kg

# Step 1: Load the street network
@st.cache_resource
def load_graph(place_name):
    G = ox.graph_from_place(place_name, network_type='walk')
    G = ox.distance.add_edge_lengths(G)

    # Convert to undirected and retain the largest connected component
    G_undirected = G.to_undirected()
    if not nx.is_connected(G_undirected):
        G_undirected = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(G_undirected)

    return G_undirected


# Step 2: Add calorie weights to the graph
def add_calorie_weights(G, user_weight):
    for u, v, data in G.edges(data=True):
        distance = data['length']  # Edge length in meters
        calories = distance * CALORIES_PER_METER_PER_KG * user_weight
        data['calories'] = calories
    return G

# Helper function to compute a single route
def compute_route(G, start_node, target_node, target_distance, max_calories):
    try:
        path = nx.shortest_path(G, source=start_node, target=target_node, weight="length")
        total_distance = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))
        total_calories = sum(G[u][v][0]['calories'] for u, v in zip(path[:-1], path[1:]))

        # Check constraints
        if abs(total_distance - target_distance) <= 0.3 * target_distance:  # ±30% tolerance
            if max_calories is None or total_calories <= max_calories:
                return (path, total_distance, total_calories)
    except nx.NetworkXNoPath:
        pass
    return None

# Optimized and Parallelized Route Finder
def find_routes_by_distance(G, start_point, target_distance, max_calories=None, max_targets=100, n_jobs=-1):
    start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])

    # Limit target nodes to nearby ones
    nearby_nodes = list(nx.ego_graph(G, start_node, radius=target_distance / 2, distance="length").nodes)[:max_targets]

    # Use parallel processing to compute routes
    routes = Parallel(n_jobs=n_jobs)(
        delayed(compute_route)(G, start_node, target_node, target_distance, max_calories) for target_node in nearby_nodes
    )

    # Filter out None values (invalid routes)
    valid_routes = [route for route in routes if route is not None]

    # Fallback logic: Return closest match if no valid routes found
    if not valid_routes:
        st.warning("No routes found matching the criteria. Returning the closest match.")
        closest_route = None
        closest_diff = float('inf')
        for target_node in G.nodes:
            try:
                path = nx.shortest_path(G, source=start_node, target=target_node, weight="length")
                total_distance = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))
                diff = abs(total_distance - target_distance)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_route = (path, total_distance, 0)  # Replace 0 with calculated calories if needed
            except nx.NetworkXNoPath:
                continue
        if closest_route:
            valid_routes.append(closest_route)

    return valid_routes

# Visualize routes on a map
def plot_routes(G, routes, start_point):
    route_map = folium.Map(location=start_point, zoom_start=14)

    # Add the start marker
    folium.Marker(start_point, popup="Start", icon=folium.Icon(color="green")).add_to(route_map)

    # Add routes
    colors = ['blue', 'red', 'purple', 'orange', 'green']
    for i, (path, distance, calories) in enumerate(routes):
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
        folium.PolyLine(route_coords, color=colors[i % len(colors)], weight=2.5, opacity=0.8).add_to(route_map)

        # Add route details
        folium.Marker(
            route_coords[-1],
            popup=f"Route {i + 1}: {distance:.2f} meters, {calories:.2f} kcal",
            icon=folium.Icon(color=colors[i % len(colors)])
        ).add_to(route_map)

    return route_map

# Streamlit app
def main():
    st.title("Interactive Running Route Planner")
    place_name = st.text_input("Enter the city name:", value="Ghent, Belgium")

    # Load the graph
    st.text("Loading the street network...")
    G = load_graph(place_name)
    st.success("Street network loaded successfully!")

    # User inputs
    user_weight = st.number_input("Enter your weight (kg):", value=70, min_value=30, max_value=200, step=1)
    target_distance = st.number_input("Enter the target distance (meters):", value=5000, min_value=1000, max_value=20000, step=100)
    max_calories = st.number_input("Enter the maximum calories (kcal, or 0 for no limit):", value=0)
    max_calories = None if max_calories == 0 else max_calories

    # Add calorie weights to the graph
    G = add_calorie_weights(G, user_weight)

    # Create the interactive map
    st.text("Select a starting point on the map:")
    center_location = ox.geocode(place_name)
    route_map = folium.Map(location=center_location, zoom_start=14)

    # Add a Folium map with click functionality
    folium.Marker(center_location, popup="Center").add_to(route_map)
    click_data = st_folium(route_map, width=700, height=500)

    # Check if a point was clicked
    if click_data and click_data.get('last_clicked'):
        lat = click_data['last_clicked']['lat']
        lon = click_data['last_clicked']['lng']
        start_point = (lat, lon)
        st.success(f"Starting point selected: Latitude {lat}, Longitude {lon}")

        # Find routes
        st.text("Finding all possible routes...")
        routes = find_routes_by_distance(G, start_point, target_distance, max_calories)

        if routes:
            st.success(f"Found {len(routes)} route(s).")
            for i, (path, distance, calories) in enumerate(routes):
                st.write(f"Route {i + 1}: {distance:.2f} meters, {calories:.2f} kcal")

            # Plot the routes
            st.text("Generating the route map...")
            route_map = plot_routes(G, routes, start_point)
            st_folium(route_map, width=700, height=500)
        else:
            st.error("No routes found matching the criteria.")
    else:
        st.info("Click on the map to select a starting point.")

if __name__ == "__main__":
    main()
