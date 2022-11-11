from machine_learning.distance_metrics import manhattan_distance
import pandas as pd

"""
The task is:
what row has the most cumulative distance between other rows?
"""
a = [[1, 1, 0],
     [0, 2, -1],
     [2, 3, 1],
     [1, 0, 4]]

df = pd.DataFrame(a,
                  columns=['P1', 'P2', 'P3'],
                  index=['A', 'B', 'C', 'D'])

max_distance = 0
max_distance_feature = ''

# Check each row
for idx, row in df.iterrows():
    feature_distance = 0
    distance_feature_name = idx

    rest_rows = df[df.index != idx]

    # Check each row in the rest data
    for rest_idx, rest_row in rest_rows.iterrows():
        distance = manhattan_distance(row, rest_row)

        feature_distance += distance
        distance_feature_name = idx

    if feature_distance > max_distance:
        max_distance = feature_distance
        max_distance_feature = idx

print("[INFO] The max feature distance: ", max_distance)
print("[INFO] The feature with max distance: ", max_distance_feature)
