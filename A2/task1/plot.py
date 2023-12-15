import matplotlib.pyplot as plt

plot_point = "plot_point.txt"
plot_file = "subgraph_plot.png"
lines = []

with open(plot_point, "r") as file:
    # Read the lines from the file
    lines = file.readlines()

#Read the data from the dat file
support = [5, 10, 25, 50, 95]
fsg_user_time = [float(val) for val in lines[0].split(' ')]
gspan_user_time = [float(val) for val in lines[1].split(' ')]
gaston_user_time = [float(val) for val in lines[2].split(' ')]

# Plot Support vs. User Time for different algorithms
plt.plot(support, fsg_user_time, marker='o', label='FSG')
plt.plot(support, gspan_user_time, marker='s', label='G-Span')
plt.plot(support, gaston_user_time, marker='^', label='Gaston')

# Set labels and title
plt.xlabel('Support (in %)')
plt.ylabel('Time (in sec)')
plt.title('Support vs. Time for Different Algorithms')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{plot_file}')
# plt.show()
