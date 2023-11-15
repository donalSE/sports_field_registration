import numpy as np


M = np.array([[0.012972800168717607, -0.003545156353409935, -6.0597331445920855],
              [0.010656455877958678, -0.15594430406836116, 87.99999999999741],
              [0.00012109608952226443, -0.0017720943644132855, 1.0]])
# Calculate its inverse
# M_inverse = np.linalg.inv(M)
# print(M_inverse)


points = {
    "px1": [633.145, 607.57, 1],
    "px2": [1242.57, 418.78499999999997, 1],
    "px3": [926, 788, 1]
}


transformed_points = {}
for key, point in points.items():
    transformed_point = np.dot(M, np.array(point))
    if transformed_point[2] != 1:
        transformed_point = transformed_point / transformed_point[2]
    transformed_points[key] = transformed_point[:2]
print(transformed_points)