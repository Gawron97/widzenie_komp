import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_skeleton(edge_index, joint_names, specific_frame=None):
    num_nodes = len(joint_names)
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i, label=joint_names[i])
    G.add_edges_from(edge_index.t().tolist())

    pos = {}
    if specific_frame is not None:
        # Prawdziwe dane (MediaPipe: odwracamy oś Y)
        for i in range(num_nodes):
            pos[i] = np.array([specific_frame[i][0], -specific_frame[i][1]])
        title = "Real Data Frame"
        offset = 0.04
    else:
        # Schemat T-Pose (uproszczony fallback)
        pos = nx.spring_layout(G)
        title = "Graph Topology"
        offset = 0.1

    # Kolory: Lewa (Niebieski), Prawa (Czerwony), Środek (Zielony)
    colors = ['#1f77b4' if "LEFT" in n else '#d62728' if "RIGHT" in n else '#2ca02c' for n in joint_names]

    fig, ax = plt.subplots(figsize=(6, 8))
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=colors, edgecolors='black')
    
    # Etykiety
    label_pos = {k: (v[0], v[1] - offset) for k, v in pos.items()}
    for k, (x, y) in label_pos.items():
        plt.text(x, y, joint_names[k], fontsize=9, ha='center', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    if specific_frame is not None:
        ax.set_aspect('equal')
        
    plt.title(title)
    plt.axis('off')
    plt.show()