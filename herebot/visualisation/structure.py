import json
import time
import sys

# Change this to your actual code2flow output filename
input_file = sys.argv[1] if len(sys.argv) > 1 else "raw.json"
output_file = "output.json"

with open(input_file) as f:
    data = json.load(f)

raw_nodes = data["graph"]["nodes"]
raw_edges = data["graph"]["edges"]

# nodes is a dict
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

# edges is a list
edges = []
if isinstance(raw_edges, dict):
    for uid, edge in raw_edges.items():
        edges.append({
            "id": uid,
            "source": edge.get("src", ""),
            "target": edge.get("dst", ""),
            "original_id": uid
        })
else:
    for edge in raw_edges:
        uid = edge.get("uid", "")
        edges.append({
            "id": uid,
            "source": edge.get("src", ""),
            "target": edge.get("dst", ""),
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
