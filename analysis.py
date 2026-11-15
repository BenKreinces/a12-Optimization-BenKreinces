
import numpy as np
import matplotlib.pyplot as plt
import landscapes
from optimization import hillClimb

starts1 = [(-1.6, -1.4), (-1.4, 1.3), (-0.5, 0),
           (0.5, -1.2), (1.4, 1.2), (1.7, -0.5)]

starts2 = [(-2.4, -2.3), (-2.3, 1.2), (0, -2.4),
           (0.1, 2.3), (2.4, 1.0), (2.3, -1.4)]

functions = [landscapes.func1, landscapes.func2]
bounds = [[-2, 2, -2, 2], [-3, 3, -3, 3]]
starts = [starts1, starts2]

s = 0.05

for j in range(2):
    f = functions[j]
    b = bounds[j]

    x, y = np.meshgrid(np.linspace(b[0], b[1], 300), np.linspace(b[2], b[3], 300))

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.contour(x, y, f(x, y), levels=20)

    for i, (x0, y0) in enumerate(starts[j]):
        path, values, n, reason = hillClimb(f, x0, y0, s, b)


        ax1.plot(path[:, 0], path[:, 1],
                 label=f"Run {i+1}")
        ax1.scatter(x0, y0, marker="o")
        ax1.scatter(*path[-1], marker="x")

        ax2.plot(values, label=f"Run {i+1}")

        print(f"Function {j+1}, Run {i+1}")
        print(f"Start: ({x0}, {y0})")
        print(f"Final: {path[-1]}")
        print(f"Value: {values[-1]:.6f}")
        print(f"Step length: {s}")
        print(f"Attempts: {n}, Reason: {reason}\n")

    ax1.set_title(f"Function {j+1}: Hill Climbing")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    ax1.legend()

    ax2.set_title(f"Function {j+1}: Objective Value")
    ax2.set_xlabel("Attempted Step")
    ax2.set_ylabel("Objective Value")
    ax2.legend()

    fig1.savefig(f"hill_paths_{j+1}.png")
    fig2.savefig(f"hill_values_{j+1}.png")

plt.show()
