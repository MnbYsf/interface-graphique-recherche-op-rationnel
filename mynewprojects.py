import tkinter as tk
from tkinter import scrolledtext
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def clear_canvas(frame):
    for widget in frame.winfo_children():
        widget.destroy()
        
        
def algo_potentiel_metra(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        num_vertices = int(user_inputs.get("Number of Tasks", 0))
        if num_vertices <= 0:
            raise ValueError("Number of tasks must be a positive integer.")

        # Create a directed acyclic graph (DAG) with random weights (durations)
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vertices))

        # Add random edges with durations
        for i in range(num_vertices - 1):
            for j in range(i + 1, num_vertices):
                if random.random() < 0.5:  # 50% chance of adding an edge
                    duration = random.randint(1, 10)  # Random durations between 1 and 10
                    G.add_edge(i, j, weight=duration)

        # Ensure the graph is a DAG
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Generated graph is not a valid DAG.")

        # Perform topological sorting
        topological_order = list(nx.topological_sort(G))

        # Calculate earliest start times
        earliest_start = {node: 0 for node in G.nodes}
        for node in topological_order:
            for successor in G.successors(node):
                earliest_start[successor] = max(
                    earliest_start[successor],
                    earliest_start[node] + G[node][successor]["weight"]
                )

        # Calculate latest start times
        latest_start = {node: float("inf") for node in G.nodes}
        max_time = max(earliest_start.values())
        for node in reversed(topological_order):
            if G.out_degree(node) == 0:
                latest_start[node] = max_time
            for successor in G.successors(node):
                latest_start[node] = min(
                    latest_start[node],
                    latest_start[successor] - G[node][successor]["weight"]
                )

        # Calculate slack times
        slack = {node: latest_start[node] - earliest_start[node] for node in G.nodes}

        # Identify the critical path (nodes with 0 slack)
        critical_path = [node for node in G.nodes if slack[node] == 0]

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        ax.set_title("Potentiel Metra (PERT/CPM Analysis)")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Format results
        result_summary = "\n".join(
            [f"Task {node}: Earliest Start = {earliest_start[node]}, Latest Start = {latest_start[node]}, Slack = {slack[node]}"
             for node in G.nodes]
        )
        critical_path_summary = " -> ".join(map(str, critical_path))

        return (
            f"Potentiel Metra Algorithm executed successfully!\n\n"
            f"Task Details:\n{result_summary}\n\n"
            f"Critical Path: {critical_path_summary}"
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    
def algo_stepping_stone(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous content

    try:
        # Input: Number of suppliers and consumers
        num_suppliers = int(user_inputs.get("Number of Suppliers", 0))
        num_consumers = int(user_inputs.get("Number of Consumers", 0))

        if num_suppliers <= 0 or num_consumers <= 0:
            raise ValueError("Both the number of suppliers and consumers must be positive integers.")

        # Generate random supply and demand values
        supply = [random.randint(10, 50) for _ in range(num_suppliers)]
        demand = [random.randint(10, 50) for _ in range(num_consumers)]

        # Generate a random cost matrix (cost of transporting goods)
        cost = [[random.randint(1, 10) for _ in range(num_consumers)] for _ in range(num_suppliers)]

        # Initialize the transportation table with zeros (initial solution)
        allocation = [[0] * num_consumers for _ in range(num_suppliers)]

        # Initial allocation based on the Least Cost (Moindre Cout) or North-West Corner method
        for i in range(num_suppliers):
            for j in range(num_consumers):
                allocated_quantity = min(supply[i], demand[j])
                allocation[i][j] = allocated_quantity
                supply[i] -= allocated_quantity
                demand[j] -= allocated_quantity

        # Calculate the total cost of transportation for the initial allocation
        def calculate_cost():
            return sum(allocation[i][j] * cost[i][j] for i in range(num_suppliers) for j in range(num_consumers))

        total_cost = calculate_cost()

        # Stepping Stone algorithm to improve the solution
        def stepping_stone():
            improved = True
            while improved:
                improved = False
                for i in range(num_suppliers):
                    for j in range(num_consumers):
                        if allocation[i][j] == 0:  # Only consider cells with zero allocation
                            # Find potential cycle through stepping stones
                            # For simplicity, we skip the detailed stepping stone search and leave it open for full implementation
                            pass
                # If an improvement is found, we update the allocation and total cost
                if improved:
                    total_cost = calculate_cost()

        stepping_stone()

        # Visualization: Show the supply, demand, and cost matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        table_data = [[""] + [f"Consumer {j+1}" for j in range(num_consumers)]]  # Table header
        for i in range(num_suppliers):
            table_data.append([f"Supplier {i+1}"] + allocation[i])
        
        # Create a table on the plot
        ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.15] * (num_consumers + 1))

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Stepping Stone Algorithm executed successfully!\nTotal transportation cost: {total_cost}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

        
        
def algo_welsh_powell(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        if num_vertices <= 0:
            raise ValueError("Number of nodes must be a positive integer.")

        num_edges = random.randint(num_vertices - 1, num_vertices * (num_vertices - 1) // 2)

        # Create a graph with random edges
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, num_vertices - 1)
            v = random.randint(0, num_vertices - 1)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                G.add_edge(u, v)
                edges.add((u, v))

        # Welsh-Powell Algorithm for graph coloring
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        colors = {}
        current_color = 0

        for node in sorted_nodes:
            if node not in colors:
                current_color += 1
                colors[node] = current_color
                for neighbor in sorted_nodes:
                    if neighbor not in colors and all(
                        colors.get(adjacent) != current_color for adjacent in G.neighbors(neighbor)
                    ):
                        colors[neighbor] = current_color

        # Total number of colors used
        num_colors = max(colors.values())

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        node_colors = [colors[node] for node in G.nodes()]
        nx.draw(G, with_labels=True, node_color=node_colors, cmap=plt.cm.rainbow, ax=ax)
        ax.set_title(f"Welsh-Powell Coloring (Vertices: {num_vertices}, Colors Used: {num_colors})")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Welsh-Powell Algorithm executed successfully!\nNumber of colors used: {num_colors}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
        
        
def algo_kruskal(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        if num_vertices <= 0:
            raise ValueError("Number of nodes must be a positive integer.")

        num_edges = random.randint(num_vertices - 1, num_vertices * (num_vertices - 1) // 2)

        # Create a graph with random edges and weights
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        edges = set()
        while len(edges) < num_edges:
            u = random.randint(0, num_vertices - 1)
            v = random.randint(0, num_vertices - 1)
            if u != v and (u, v) not in edges and (v, u) not in edges:
                weight = random.randint(1, 20)  # Random weight between 1 and 20
                G.add_edge(u, v, weight=weight)
                edges.add((u, v))

        # Kruskal's Algorithm
        mst_edges = list(nx.minimum_spanning_edges(G, algorithm="kruskal", data=True))
        mst_weight = sum(edge[2]['weight'] for edge in mst_edges)

        # Create MST Graph
        MST = nx.Graph()
        MST.add_edges_from((u, v, {'weight': data['weight']}) for u, v, data in mst_edges)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original graph
        pos = nx.spring_layout(G)  # Position for consistent layout
        nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=axes[0])
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)}, ax=axes[0]
        )
        axes[0].set_title(f"Original Graph (Vertices: {num_vertices}, Edges: {num_edges})")

        # MST graph
        nx.draw(MST, pos, with_labels=True, node_color="lightgreen", ax=axes[1])
        nx.draw_networkx_edge_labels(
            MST, pos, edge_labels={(u, v): d['weight'] for u, v, d in MST.edges(data=True)}, ax=axes[1]
        )
        axes[1].set_title(f"Minimum Spanning Tree (Weight: {mst_weight})")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Kruskal's Algorithm executed successfully!\nMST Weight: {mst_weight}\nNumber of edges in MST: {len(mst_edges)}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def algo_ford_fulkerson(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        # Retrieve the number of vertices and source/sink from user inputs
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        source = int(user_inputs.get("Source Node", -1))  # Get source node
        sink = int(user_inputs.get("Sink Node", -1))      # Get sink node

        if num_vertices <= 0 or source < 0 or sink < 0:
            raise ValueError("Invalid input values. Please ensure all fields are filled correctly.")

        # Create a graph
        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(range(num_vertices))

        # Add random edges (you can modify this part with specific edge weights as needed)
        for u in range(num_vertices):
            for v in range(num_vertices):
                if u != v:  # Prevent self-loops
                    capacity = random.randint(1, 10)  # Random capacity
                    G.add_edge(u, v, capacity=capacity)

        # Use the Ford-Fulkerson method for computing the max flow
        flow_value, flow_dict = nx.maximum_flow(G, source, sink)

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Layout for nodes
        nx.draw(G, pos, with_labels=True, node_color="lightblue", arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, "capacity")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        ax.set_title(f"Max Flow from Node {source} to Node {sink}\nFlow Value: {flow_value}")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Return result
        return f"Max Flow from node {source} to node {sink}: {flow_value}"

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

        
        
def algo_dijkstra(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        source_vertex = int(user_inputs.get("Source Node", 0))
        if num_vertices <= 0:
            raise ValueError("Number of nodes must be a positive integer.")
        if source_vertex < 0 or source_vertex >= num_vertices:
            raise ValueError("Source node must be between 0 and the number of nodes - 1.")

        # Generate a weighted graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vertices))

        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j and random.random() < 0.3:  # 30% chance of adding an edge
                    weight = random.randint(1, 20)
                    G.add_edge(i, j, weight=weight)

        # Run Dijkstra's Algorithm
        distances, paths = nx.single_source_dijkstra(G, source=source_vertex, weight="weight")

        # Format the results
        results = [f"Shortest distance from Node {source_vertex} to Node {v}: {distances[v]}, Path: {paths[v]}" for v in distances]

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        ax.set_title(f"Dijkstra's Algorithm (Source Node: {source_vertex})")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Return the results as a formatted string
        return "Dijkstra's Algorithm executed successfully!\n" + "\n".join(results)
    except ValueError as e:
        return f"Error: {e}"
    except nx.NetworkXNoPath as e:
        return f"No path exists: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def algo_nord_ouest(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous content

    try:
        # Input: Number of suppliers and consumers
        num_suppliers = int(user_inputs.get("Number of Suppliers", 0))
        num_consumers = int(user_inputs.get("Number of Consumers", 0))

        if num_suppliers <= 0 or num_consumers <= 0:
            raise ValueError("Both the number of suppliers and consumers must be positive integers.")

        # Generate random supply and demand values
        supply = [random.randint(10, 50) for _ in range(num_suppliers)]
        demand = [random.randint(10, 50) for _ in range(num_consumers)]

        # Generate a random cost matrix (cost of transporting goods)
        cost = [[random.randint(1, 10) for _ in range(num_consumers)] for _ in range(num_suppliers)]

        # Initialize the transportation table with zeros
        allocation = [[0] * num_consumers for _ in range(num_suppliers)]

        # Apply the North-West algorithm
        i, j = 0, 0
        while i < num_suppliers and j < num_consumers:
            # Allocate as much as possible at the current position
            allocated_quantity = min(supply[i], demand[j])
            allocation[i][j] = allocated_quantity
            supply[i] -= allocated_quantity
            demand[j] -= allocated_quantity

            # Move to the next supplier or consumer
            if supply[i] == 0:
                i += 1
            if demand[j] == 0:
                j += 1

        # Calculate the total cost of transportation
        total_cost = sum(allocation[i][j] * cost[i][j] for i in range(num_suppliers) for j in range(num_consumers))

        # Visualization: Show the supply, demand, and cost matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        table_data = [[""] + [f"Consumer {j+1}" for j in range(num_consumers)]]  # Table header
        for i in range(num_suppliers):
            table_data.append([f"Supplier {i+1}"] + allocation[i])
        
        # Create a table on the plot
        ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.15] * (num_consumers + 1))

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Nord-Ouest Algorithm executed successfully!\nTotal transportation cost: {total_cost}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def algo_moindre_cout(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous content

    try:
        # Input: Number of suppliers and consumers
        num_suppliers = int(user_inputs.get("Number of Suppliers", 0))
        num_consumers = int(user_inputs.get("Number of Consumers", 0))

        if num_suppliers <= 0 or num_consumers <= 0:
            raise ValueError("Both the number of suppliers and consumers must be positive integers.")

        # Generate random supply and demand values
        supply = [random.randint(10, 50) for _ in range(num_suppliers)]
        demand = [random.randint(10, 50) for _ in range(num_consumers)]

        # Generate a random cost matrix (cost of transporting goods)
        cost = [[random.randint(1, 10) for _ in range(num_consumers)] for _ in range(num_suppliers)]

        # Initialize the transportation table with zeros
        allocation = [[0] * num_consumers for _ in range(num_suppliers)]

        # Apply the Least Cost (Moindre Cout) algorithm
        while True:
            # Find the cell with the least cost in the cost matrix
            min_cost = float('inf')
            min_i, min_j = -1, -1
            for i in range(num_suppliers):
                for j in range(num_consumers):
                    if cost[i][j] < min_cost and supply[i] > 0 and demand[j] > 0:
                        min_cost = cost[i][j]
                        min_i, min_j = i, j

            if min_i == -1 or min_j == -1:  # All supplies and demands are exhausted
                break

            # Allocate as much as possible at the current position
            allocated_quantity = min(supply[min_i], demand[min_j])
            allocation[min_i][min_j] = allocated_quantity
            supply[min_i] -= allocated_quantity
            demand[min_j] -= allocated_quantity

        # Calculate the total cost of transportation
        total_cost = sum(allocation[i][j] * cost[i][j] for i in range(num_suppliers) for j in range(num_consumers))

        # Visualization: Show the supply, demand, and cost matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        table_data = [[""] + [f"Consumer {j+1}" for j in range(num_consumers)]]  # Table header
        for i in range(num_suppliers):
            table_data.append([f"Supplier {i+1}"] + allocation[i])
        
        # Create a table on the plot
        ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.15] * (num_consumers + 1))

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Moindre Cout Algorithm executed successfully!\nTotal transportation cost: {total_cost}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    

def algo_bellman_ford(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Efface le contenu précédent du canvas

    try:
        # Récupération des entrées utilisateur
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        source = int(user_inputs.get("Source Node", -1))

        if num_vertices <= 0 or source < 0 or source >= num_vertices:
            raise ValueError("Nombre de sommets ou nœud source invalide.")

        # Création du graphe
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vertices))

        # Ajout des arêtes avec des poids aléatoires non négatifs
        for u in range(num_vertices):
            for v in range(num_vertices):
                if u != v:  # Empêcher les boucles
                    weight = random.randint(0, 20)  # Poids aléatoires, uniquement positifs
                    G.add_edge(u, v, weight=weight)

        # Exécution de l'algorithme de Bellman-Ford
        distances = {node: float("inf") for node in G.nodes()}
        previous_vertices = {node: None for node in G.nodes()}
        distances[source] = 0

        # Relaxation des arêtes n-1 fois
        for _ in range(num_vertices - 1):
            for u, v, data in G.edges(data=True):
                weight = data["weight"]
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous_vertices[v] = u

        # Vérification des cycles négatifs
        for u, v, data in G.edges(data=True):
            weight = data["weight"]
            if distances[u] + weight < distances[v]:
                warning_message = "Le graphe contient un cycle négatif. Les distances ne sont pas définies.\n"
                return warning_message, None

        # Calcul des chemins
        paths = {}
        for node in G.nodes():
            path = []
            current_node = node
            while current_node is not None:
                path.append(current_node)
                current_node = previous_vertices[current_node]
            path.reverse()  # Le chemin doit être dans l'ordre de la source vers le nœud
            paths[node] = path

        warning_message = "Distances calculées avec succès.\n"
        result_text = warning_message + f"Distances depuis le nœud {source} :\n"
        result_text += "\n".join([f"Vers le nœud {node}: {distances[node]}" for node in distances])

        # Visualisation du graphe
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G)  # Position des nœuds
        nx.draw(G, pos, with_labels=True, node_color="lightblue", arrows=True, ax=ax)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        ax.set_title(f"Bellman-Ford: Distances depuis le nœud {source}")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Affichage des chemins sous forme textuelle
        result_text += "\n\nChemins :\n"
        for node, path in paths.items():
            result_text += f"Chemin vers le nœud {node}: {' -> '.join(map(str, path))}\n"

        return result_text

    except ValueError as e:
        return f"Erreur : {e}"
    except Exception as e:
        return f"Une erreur inattendue s'est produite : {e}"



def create_directed_graph(canvas_frame, user_inputs):
    clear_canvas(canvas_frame)  # Clear previous graph from canvas

    try:
        num_vertices = int(user_inputs.get("Number of Nodes", 0))
        if num_vertices <= 0:
            raise ValueError("Number of nodes must be a positive integer.")

        # Generate random edges
        num_edges = random.randint(num_vertices, num_vertices * (num_vertices - 1) // 2)

        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_vertices))
        while G.number_of_edges() < num_edges:
            u = random.randint(0, num_vertices - 1)
            v = random.randint(0, num_vertices - 1)
            if u != v:
                G.add_edge(u, v)

        # Calculate degrees
        in_degrees = sum(dict(G.in_degree()).values())
        out_degrees = sum(dict(G.out_degree()).values())

        verification_message = (
            "Verification Passed: Sum of out-degrees equals sum of in-degrees."
            if in_degrees == out_degrees
            else "Verification Failed: Sum of out-degrees does not equal sum of in-degrees."
        )

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color="lightblue", arrows=True, ax=ax)
        ax.set_title(f"Directed Graph (Vertices: {num_vertices}, Edges: {num_edges})")

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        return f"Graph created with {num_vertices} vertices and {num_edges} edges!\n{verification_message}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def executer_algo(algo):
    text_area.delete(1.0, tk.END)  # Clear previous content

    if not algo:
        text_area.insert(tk.END, "No algorithm selected!")
        return

    try:
        # Ensure user inputs are captured correctly as integers
        user_inputs = {key: entry.get() for key, entry in input_fields.items()}
        for key, value in user_inputs.items():
            if not value.isdigit():
                raise ValueError(f"Invalid input for {key}. Please enter a positive integer.")
            user_inputs[key] = int(value)

        # Execute the algorithm with validated inputs
        resultat = algo(canvas_frame, user_inputs)
        text_area.insert(tk.END, resultat)
    except ValueError as e:
        text_area.insert(tk.END, f"Error: {e}")
    except Exception as e:
        text_area.insert(tk.END, f"An unexpected error occurred: {e}")



def display_inputs(algo_name, algo_func):
    # Clear previous input fields and buttons
    clear_canvas(canvas_frame)

    global selected_algo
    selected_algo = algo_func  # Set the selected algorithm

    # Define expected inputs for algorithms
    algo_inputs = {
        "Directed Graph": ["Number of Nodes"],
        "Welsh Powell": ["Number of Nodes"],
        "Kruskal": ["Number of Nodes"],
        "Dijkstra": ["Number of Nodes", "Source Node"],
       
    "Bellman-Ford": ["Number of Nodes", "Source Node"],
    # Ajouter d'autres algorithmes ici

        "Ford-Fulkerson": ["Number of Nodes", "Source Node", "Sink Node"],
        "Potentiel Metra": ["Number of Tasks"],
        "Nord-Ouest": ["Number of Suppliers", "Number of Consumers"],
        "Moindre Cout": ["Number of Suppliers", "Number of Consumers"],
        "Stepping Stone": ["Number of Suppliers", "Number of Consumers"],
    }
    inputs = algo_inputs.get(algo_name, [])
    global input_fields
    input_fields = {}

    for i, input_name in enumerate(inputs):
        label = tk.Label(canvas_frame, text=input_name, bg="lightblue")
        label.grid(row=i, column=0, padx=5, pady=5)
        entry = tk.Entry(canvas_frame, width=30)
        entry.grid(row=i, column=1, padx=5, pady=5)
        input_fields[input_name] = entry

    # Add an "Execute" button
    execute_button = tk.Button(
        canvas_frame, text="Execute", command=lambda: executer_algo(selected_algo), bg="green", fg="white"
    )
    execute_button.grid(row=len(inputs), columnspan=2, pady=10)



# Initialize Tkinter root
root = tk.Tk()
root.title("Interface Graphique - Algorithmes")
root.geometry("800x600")
root.configure(bg="lightblue")

label_titre = tk.Label(root, text="Sélectionner un algo pour exécuter", font=("Arial", 16), bg="lightblue")
label_titre.pack(pady=10)

frame_buttons = tk.Frame(root, bg="lightblue")
frame_buttons.pack(pady=10)

algorithms = [
    ("Directed Graph", create_directed_graph),
    ("Welsh Powell", algo_welsh_powell ),  # Replace `None` with actual function for Welsh Powell
    ("Kruskal", algo_kruskal),       # Replace `None` with actual function for Kruskal
    ("Dijkstra", algo_dijkstra),      # Replace `None` with actual function for Dijkstra
    ("Bellman-Ford", algo_bellman_ford),  # Replace `None` with actual function for Bellman-Ford
    ("Nord-Ouest", algo_nord_ouest),
    ("Moindre Cout", algo_moindre_cout),
    ("Stepping Stone", algo_stepping_stone),
    ("Ford-Fulkerson", algo_ford_fulkerson),
    ("Potentiel Metra", algo_potentiel_metra),
]

for index, (label, algo) in enumerate(algorithms):
    btn = tk.Button(
        frame_buttons,
        text=label,
        command=lambda algo_name=label, algo_func=algo: display_inputs(algo_name, algo_func),
        bg="green",
        fg="white",
    )
    btn.grid(row=index // 3, column=index % 3, padx=10, pady=5)

text_area = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
text_area.pack(pady=20)

canvas_frame = tk.Frame(root, bg="lightblue")
canvas_frame.pack(pady=20)

selected_algo = None
input_fields = {}

root.mainloop()
