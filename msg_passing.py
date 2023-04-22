import numpy as np
from scipy.linalg import sqrtm  # numpy doesn't support sqrt of a matrix
from scipy.special import softmax  # note: softmax turns input into probabilities

import networkx as nx  # used to handle graph data
from networkx.algorithms.community.modularity_max import greedy_modularity_communities

import matplotlib.pyplot as plt
from matplotlib import animation


# defines the graph connections
# A is the adjacency matrix

adjacency_matrix = np.array(
    [  #  1  2  3  4  5
        [0, 1, 0, 0, 0],  # 1
        [1, 0, 1, 0, 0],  # 2
        [0, 1, 0, 1, 1],  # 3
        [0, 0, 1, 0, 0],  # 4
        [0, 0, 1, 0, 0],  # 5
    ]
)

# + 1 to the matrix so that it goes from 1 to n instead of 0 to n - 1
feature_vector = (np.arange(adjacency_matrix.shape[0]) + 1).reshape(
    -1, 1
)  # convert the vector to column vector

# -------- SUM OF FEATURE VECTOR

# H_sum = feature_vector
# OLD_H_sum = np.zeros(H_sum.shape[1])
# iterations = 0

# while not (H_sum == OLD_H_sum).all():
#     OLD_H_sum = H_sum.copy()
#     H_sum = adjacency_matrix @ H_sum

#     # interesting note: during this calculation
#     # any value in H can become negative this is because
#     # of the formula for dot product that uses cosine of theta

#     # # remove comments to check H_sum values
#     # print(f"iter {iterations + 1}")
#     # print(H.reshape(1, -1))
#     # print()

#     iterations += 1

# -------- AVG OF FEATURE VECTOR

degree_matrix = np.zeros(adjacency_matrix.shape)
np.fill_diagonal(degree_matrix, adjacency_matrix.sum(axis=0))

degree_matrix_inv = np.linalg.inv(degree_matrix)

# interesting property of avg matrix
# sum of the rows of the matrix will be equal to 1
# (at least as close as floating numbers allow it)
avg_adj_matrix = degree_matrix_inv @ adjacency_matrix

H_avg = feature_vector

# for iterations in range(50):
#     H_avg = avg_adj_matrix @ H_avg

# # remove comments to check H_avg values
# print(f"iter {iterations + 1}")
# print(H_avg.reshape(1, -1))
# print()

# -------- NORMALIZATION

graph = nx.from_numpy_array(adjacency_matrix)

adj_identity = np.eye(graph.number_of_nodes())  # create the identity matrix
adj_tild = adjacency_matrix + adj_identity


# the reason for adding the identity matrix and the adjacent matrix
# is to create the self connection between the nodes and itself
# its like telling the node to always remember who you are
# while also learning from the neighboring nodes

degree_matrix_tild = np.zeros_like(adj_tild)
np.fill_diagonal(degree_matrix_tild, adj_tild.sum(axis=1).flatten())

degree_matrix_tild_inv_sqrt = np.linalg.inv(sqrtm(degree_matrix_tild))

adjacent_normalized_matrix = (
    degree_matrix_tild_inv_sqrt @ adj_tild @ degree_matrix_tild_inv_sqrt
)

# -------- NORMALIZATION PLOTTING & VISUALIZATION

node_labels = {i: i + 1 for i in range(graph.number_of_nodes())}
pos = nx.layout.spectral_layout(graph)

fig, ax = plt.subplots(figsize=(10, 10))

nx.draw(
    graph,
    pos,
    with_labels=True,
    labels=node_labels,
    node_color="#83C167",
    ax=ax,
    edge_color="gray",
    node_size=1500,
    font_size=30,
    font_family="serif",
)

plt.savefig("images/simple_graph.png", bbox_inches="tight", transparent=True)

H = np.zeros((graph.number_of_nodes(), 1))
H[0, 0] = 1  # the "water drop"

results = [H.flatten()]

OLD_H = np.zeros(H.shape[1])
iterations = 0

while not (H == OLD_H).all():
    OLD_H = H.copy()
    H = adjacent_normalized_matrix @ H
    results.append(H.flatten())

    # interesting note: during this calculation
    # any value in H can become negative this is because
    # of the formula for dot product that uses cosine of theta

    # # remove comments to check H values
    # print(f"iter {iterations + 1}")
    # print(H.flatten() - OLD_H.flatten())
    # print(H.flatten())
    # print()

    iterations += 1

# -------- VIDEO FORMAT

fig, ax = plt.subplots(figsize=(10, 10))

kwargs = {
    "cmap": "hot",
    "node_size": 1500,
    "edge_color": "gray",
    "vmin": np.array(results).min(),
    "vmax": np.array(results).max() * 1.1,
}


def update(idx):
    ax.clear()
    colors = results[idx]
    nx.draw(graph, pos, node_color=colors, ax=ax, **kwargs)
    ax.set_title(f"Iter={idx}", fontsize=20)


anim = animation.FuncAnimation(
    fig, update, frames=len(results), interval=50, repeat=True
)

anim.save(
    "images/gnn-basics.gif",
    dpi=600,
    bitrate=-1,
    savefig_kwargs={"transparent": False, "facecolor": "none"},
)
