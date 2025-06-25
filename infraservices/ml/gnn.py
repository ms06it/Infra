import networkx as nx
import matplotlib.pyplot as plt

# Create a simple infrastructure dependency graph
G = nx.Graph()
G.add_edges_from([
    ('Service A', 'Service B'),
    ('Service A', 'Service C'),
    ('Service B', 'Service D'),
    ('Service C', 'Service D')
])

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
plt.title("Infrastructure Dependency Graph")
plt.savefig('gnn_graph.png')
plt.show()

