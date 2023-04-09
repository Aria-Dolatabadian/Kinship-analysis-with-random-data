import numpy as np
import matplotlib.pyplot as plt

# Kinship matrix
kinship = np.array([[1.0, 0.4, 0.3, 0.1],
                    [0.4, 1.0, 0.2, 0.1],
                    [0.3, 0.2, 1.0, 0.5],
                    [0.1, 0.1, 0.5, 1.0]])

# Plot the kinship matrix
fig, ax = plt.subplots()
im = ax.imshow(kinship, cmap='coolwarm')

# Set the x and y tick labels
ax.set_xticks(np.arange(len(kinship)))
ax.set_yticks(np.arange(len(kinship)))
ax.set_xticklabels(["Indiv 1", "Indiv 2", "Indiv 3", "Indiv 4"])
ax.set_yticklabels(["Indiv 1", "Indiv 2", "Indiv 3", "Indiv 4"])

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Set the title and colorbar
ax.set_title("Kinship Matrix")
fig.colorbar(im)

# Show the plot
plt.show()
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# Kinship matrix
kinship = np.array([[1.0, 0.4, 0.3, 0.1],
                    [0.4, 1.0, 0.2, 0.1],
                    [0.3, 0.2, 1.0, 0.5],
                    [0.1, 0.1, 0.5, 1.0],])

# Compute linkage matrix
linkage_matrix = hierarchy.linkage(kinship, method='ward')

# Plot the kinship matrix and dendrogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im = ax1.imshow(kinship, cmap='coolwarm')

# Set the x and y tick labels
ax1.set_xticks(np.arange(len(kinship)))
ax1.set_yticks(np.arange(len(kinship)))
ax1.set_xticklabels(["Indiv 1", "Indiv 2", "Indiv 3", "Indiv 4"])
ax1.set_yticklabels(["Indiv 1", "Indiv 2", "Indiv 3", "Indiv 4"])

# Rotate the tick labels and set their alignment
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Set the title and colorbar
ax1.set_title("Kinship Matrix")
fig.colorbar(im, ax=ax1)

# Plot the dendrogram
hierarchy.dendrogram(linkage_matrix, ax=ax2, labels=["Indiv 1", "Indiv 2", "Indiv 3", "Indiv 4"])

# Set the title and labels
ax2.set_title("Dendrogram")
ax2.set_xlabel("Individuals")
ax2.set_ylabel("Distance")

# Show the plot
plt.show()
#
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# Generate kinship data
kinship = np.random.uniform(low=0.0, high=1.0, size=(50, 50))
kinship = (kinship + kinship.T) / 2.0  # Ensure symmetric matrix
np.fill_diagonal(kinship, 1.0)  # Ensure diagonal is 1.0

# Compute linkage matrix
linkage_matrix = hierarchy.linkage(kinship, method='ward')

# Plot the kinship matrix and dendrogram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im = ax1.imshow(kinship, cmap='coolwarm')

# Set the x and y tick labels and markers
xticks = np.arange(0, 50, 4)
yticks = np.arange(0, 50, 4)
markersize = 20
ax1.set_xticks(xticks)
ax1.set_yticks(yticks)
ax1.set_xticklabels([f"Indiv {i+1}" for i in xticks])
ax1.set_yticklabels([f"Indiv {i+1}" for i in yticks])
ax1.tick_params(axis='both', which='both', length=0)  # Hide tick marks
ax1.scatter(x=np.repeat(xticks, len(yticks)), y=np.tile(yticks, len(xticks)),
            marker='s', s=markersize, color='black', alpha=0.5)

# Rotate the x tick labels and set their alignment
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Set the title and colorbar
ax1.set_title("Kinship Matrix")
fig.colorbar(im, ax=ax1)

# Plot the dendrogram
hierarchy.dendrogram(linkage_matrix, ax=ax2, labels=[f"Indiv {i+1}" for i in range(50)])

# Set the title and labels
ax2.set_title("Dendrogram")
ax2.set_xlabel("Individuals")
ax2.set_ylabel("Distance")

# Show the plot
plt.show()


