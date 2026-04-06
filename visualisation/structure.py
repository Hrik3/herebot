import json
import time
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else "raw.json"
output_file = "output.json"

with open(input_file) as f:
    data = json.load(f)

raw_nodes = data["graph"]["nodes"]
raw_edges = data["graph"]["edges"]

# --- Nodes (dict in this file) ---
nodes = []
if isinstance(raw_nodes, dict):
    for uid, node in raw_nodes.items():
        nodes.append({
            "id": uid,
            "label": node.get("label", ""),
            "name": node.get("name", ""),
            "type": "function",
            "original_id": uid
        })
else:
    for node in raw_nodes:
        uid = node.get("uid", "")
        nodes.append({
            "id": uid,
            "label": node.get("label", ""),
            "name": node.get("name", ""),
            "type": "function",
            "original_id": uid
        })

# --- Edges (list in this file, keys are "source"/"target", no "uid") ---
edges = []
if isinstance(raw_edges, dict):
    for uid, edge in raw_edges.items():
        # dict format: try src/dst first, fall back to source/target
        edges.append({
            "id": uid,
            "source": edge.get("src") or edge.get("source", ""),
            "target": edge.get("dst") or edge.get("target", ""),
            "original_id": uid
        })
else:
    for idx, edge in enumerate(raw_edges):
        # list format: try source/target first, fall back to src/dst
        source = edge.get("source") or edge.get("src", "")
        target = edge.get("target") or edge.get("dst", "")
        uid = edge.get("uid") or edge.get("id") or f"edge_{idx}"
        edges.append({
            "id": uid,
            "source": source,
            "target": target,
            "original_id": uid
        })

structured = {
    "metadata": {
        "source": "code2flow",
        "original_file": input_file,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "conversion_timestamp": str(time.time())
    },
    "nodes": nodes,
    "edges": edges
}

with open(output_file, "w") as f:
    json.dump(structured, f, indent=2)

print(f"Done! {len(nodes)} nodes, {len(edges)} edges -> {output_file}")