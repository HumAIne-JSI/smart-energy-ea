import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt

# Load pandapower network
net = pp.from_json("C:\\Users\\Gasper\\Documents\\Projekti\\IJS\\smart-energy-ea\\data\\digital_twin_ext_grid.json")

# Draw simple topology
fig, ax = plt.subplots(figsize=(14, 10))

plot.simple_plot(
    net,
    ax=ax,
    show_plot=False,
    bus_size=0.08,
    line_width=1.0
)

plt.title("Pandapower Grid Topology")
plt.savefig("grid_topology.png", dpi=300, bbox_inches="tight")
plt.show()