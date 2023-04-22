import numpy as np

adjacency_matrix = np.array(
    [  #  1  2  3  4  5
        [0, 1, 0, 0, 0],  # 1
        [1, 0, 1, 0, 0],  # 2
        [0, 1, 0, 1, 1],  # 3
        [0, 0, 1, 0, 0],  # 4
        [0, 0, 1, 0, 0],  # 5
    ]
)

feature_vector = np.array(
    [411525376, 1425715712, 993510144, 1008133248, 1008133248]
).reshape(-1, 1)
final_vector = np.array(
    [1425715712, 1405035520, -852985088, 993510144, 993510144]
).reshape(-1, 1)

print(adjacency_matrix)
print(feature_vector)

print()

result_vector = adjacency_matrix @ feature_vector
print(result_vector)
print((result_vector == final_vector).all())
