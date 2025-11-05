#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy similarity remover visualizer (Matplotlib interactive version)

At each step, the node with the highest degree (most similar connections)
is removed until no edges remain.
You can interactively step through the process using Matplotlib buttons.
"""

import math
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Optional dependency: networkx (for nicer graph layouts)
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False


# ---------- Core algorithm ----------
def greedy_order_by_degree(edges: List[Tuple[str, str]]):
    """Return the removal order and snapshots of the graph state."""
    nodes = set()
    for u, v in edges:
        if u == v:
            continue
        nodes.add(u)
        nodes.add(v)

    adj = {u: set() for u in nodes}
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)

    def edges_now():
        out = []
        for u in adj:
            for v in adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    snapshots = []
    removed = []
    step = 0

    def has_edges():
        return any(len(nei) > 0 for nei in adj.values())

    while has_edges():
        # Choose node with the largest degree (tie → lexicographically smallest)
        cand = None
        max_deg = -1
        for u in sorted(adj.keys()):
            deg = len(adj[u])
            if deg > max_deg:
                max_deg = deg
                cand = u

        # Save a snapshot before removing
        snapshots.append({
            "step": step,
            "nodes": sorted(adj.keys()),
            "edges": edges_now(),
            "next_remove": (cand, max_deg)
        })
        step += 1

        # Remove the chosen node
        for v in list(adj[cand]):
            adj[v].discard(cand)
        del adj[cand]
        removed.append((cand, max_deg))

    # Final snapshot (no edges left)
    snapshots.append({"step": step, "nodes": sorted(adj.keys()), "edges": [], "next_remove": None})
    return removed, snapshots


def layout_positions(nodes: List[str], edges: List[Tuple[str, str]]):
    """Compute node positions using spring layout (if networkx available) or circular fallback."""
    if HAVE_NX:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return nx.spring_layout(G, seed=42)
    # Circle layout fallback
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        ang = 2 * math.pi * i / max(1, n)
        pos[node] = (math.cos(ang), math.sin(ang))
    return pos


# ---------- Interactive visualization ----------
class Stepper:
    """Matplotlib interactive stepper: click Prev / Next to go through removal steps."""
    def __init__(self, edges: List[Tuple[str, str]]):
        removed, snaps = greedy_order_by_degree(edges)
        self.snaps = snaps
        self.idx = 0
        all_nodes = sorted({n for s in snaps for n in s["nodes"]})
        base_edges = snaps[0]["edges"] if snaps and snaps[0]["edges"] else edges
        self.pos = layout_positions(all_nodes, base_edges)

        # Create Matplotlib figure and buttons
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        axprev = plt.axes([0.22, 0.05, 0.15, 0.075])
        axnext = plt.axes([0.63, 0.05, 0.15, 0.075])
        self.bprev = Button(axprev, 'Prev')
        self.bnext = Button(axnext, 'Next')
        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        self.draw()

    def draw(self):
        """Redraw the current snapshot."""
        self.ax.clear()
        s = self.snaps[self.idx]
        nodes, edges, to_remove, step = s["nodes"], s["edges"], s["next_remove"], s["step"]

        title = f"Step {step}"
        if to_remove is not None:
            title += f" — Next remove: {to_remove[0]} (degree {to_remove[1]})"
        else:
            title += " — Done (no edges)"
        self.ax.set_title(title)

        # Draw edges
        for (u, v) in edges:
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.ax.plot([x1, x2], [y1, y2])

        # Draw nodes (larger for the next-to-remove node)
        for n in nodes:
            x, y = self.pos[n]
            size = 400 if (to_remove is not None and n == to_remove[0]) else 120
            self.ax.scatter([x], [y], s=size)
            self.ax.text(x, y, n)

        self.ax.axis('equal')
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def next(self, event=None):
        """Go to next step."""
        self.idx = min(self.idx + 1, len(self.snaps) - 1)
        self.draw()

    def prev(self, event=None):
        """Go to previous step."""
        self.idx = max(self.idx - 1, 0)
        self.draw()


# ---------- Example ----------
if __name__ == "__main__":
    # Create a random similarity graph for demonstration
    random.seed(5)
    N = 12
    p = 0.22
    names = [f"img{i}" for i in range(N)]
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if random.random() < p:
                edges.append((names[i], names[j]))
    if not edges:
        edges.append((names[0], names[1]))

    # Launch the interactive visualizer
    Stepper(edges)
    plt.show()
