import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

def _default_human_pose(joint_names):
    idx = {n: i for i, n in enumerate(joint_names)}
    pos = {}

    def put(name, x, y):
        if name in idx:
            pos[idx[name]] = np.array([x, y], dtype=np.float32)

    # Głowa
    put("NOSE", 0.0, 1.00)
    put("LEFT_EYE", -0.05, 0.98)
    put("RIGHT_EYE", 0.05, 0.98)
    put("LEFT_EAR", -0.12, 0.95)
    put("RIGHT_EAR", 0.12, 0.95)

    # Barki / ręce
    put("LEFT_SHOULDER", -0.20, 0.80)
    put("RIGHT_SHOULDER", 0.20, 0.80)
    put("LEFT_ELBOW", -0.35, 0.65)
    put("RIGHT_ELBOW", 0.35, 0.65)
    put("LEFT_WRIST", -0.45, 0.50)
    put("RIGHT_WRIST", 0.45, 0.50)

    # Biodra / nogi
    put("LEFT_HIP", -0.15, 0.55)
    put("RIGHT_HIP", 0.15, 0.55)
    put("LEFT_KNEE", -0.15, 0.32)
    put("RIGHT_KNEE", 0.15, 0.32)
    put("LEFT_ANKLE", -0.15, 0.05)
    put("RIGHT_ANKLE", 0.15, 0.05)

    # Wszystkie nieustawione — w kolumnie przy środku
    unset = [i for i in range(len(joint_names)) if i not in pos]
    if unset:
        ys = np.linspace(0.75, 0.15, num=len(unset), dtype=np.float32)
        for k, i in enumerate(unset):
            pos[i] = np.array([0.0, float(ys[k])], dtype=np.float32)

    return pos

def visualize_skeleton_human(
    edge_index,
    joint_names,
    specific_frame=None,      # (V, 2) albo None
    title=None,
    show_labels=True,
    node_size=320,
    edge_width=2,
    normalize=True,           # centrowanie + skalowanie dla czytelności
):
    num_nodes = len(joint_names)

    # edge_index -> CPU
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu()
    edges = edge_index.t().tolist()

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # --- POS ---
    if specific_frame is not None:
        frame = np.asarray(specific_frame, dtype=np.float32)
        if frame.shape != (num_nodes, 2):
            raise ValueError(f"specific_frame musi mieć shape ({num_nodes}, 2), a ma {frame.shape}")

        xy = frame.copy()

        # odwróć Y (MediaPipe ma Y w dół ekranu)
        xy[:, 1] = -xy[:, 1]

        if normalize:
            # centrowanie
            xy = xy - xy.mean(axis=0, keepdims=True)
            # skalowanie do przyjemnego rozmiaru (unikamy dzielenia przez 0)
            scale = np.max(np.linalg.norm(xy, axis=1))
            if scale > 1e-8:
                xy = xy / scale

        pos = {i: xy[i] for i in range(num_nodes)}
        ttl = title or "Human Skeleton (from data frame)"
    else:
        pos = _default_human_pose(joint_names)
        ttl = title or "Human Skeleton (template pose)"

    # --- KOLORY ---
    colors = [
        '#1f77b4' if "LEFT" in n else
        '#d62728' if "RIGHT" in n else
        '#2ca02c'
        for n in joint_names
    ]

    fig, ax = plt.subplots(figsize=(7, 9))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=edge_width, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=colors, edgecolors='black')

    if show_labels:
        for i in range(num_nodes):
            x, y = pos[i]
            ax.text(
                x, y - 0.03,
                joint_names[i],
                fontsize=8,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
            )

    ax.set_title(ttl)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()