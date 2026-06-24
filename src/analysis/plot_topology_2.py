import pandapower as pp
import matplotlib.pyplot as plt
import networkx as nx

# Load pandapower network
net = pp.from_json("C:\\Users\\Gasper\\Documents\\Projekti\\IJS\\smart-energy-ea\\data\\digital_twin_ext_grid.json")

# Build graph from buses and lines
G = nx.Graph()

for bus_idx, bus in net.bus.iterrows():
    G.add_node(bus_idx, label=str(bus["name"]))

for line_idx, line in net.line.iterrows():
    if bool(line["in_service"]):
        from_bus = int(line["from_bus"])
        to_bus = int(line["to_bus"])
        G.add_edge(from_bus, to_bus, label=str(line["name"]))

# Use real/geographical coordinates if available
if hasattr(net, "bus_geodata") and not net.bus_geodata.empty:
    pos = {
        int(idx): (row["x"], row["y"])
        for idx, row in net.bus_geodata.iterrows()
    }
else:
    pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(16, 12))

nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.7)
nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_labels(
    G,
    pos,
    labels=nx.get_node_attributes(G, "label"),
    font_size=8
)

plt.title("Digital Twin Grid Topology – Buses and Transmission Lines")
plt.axis("off")
plt.tight_layout()
plt.savefig("digital_twin_topology.png", dpi=300)
plt.show()