
import numpy as np
import matplotlib.pyplot as plt
import landscapes
from optimization import hillClimb, simAnnealing, randomSampling


if __name__ == "__main__":
    #Random Sampling
    x1, y1, vals1, bestx1, besty1, best1 = randomSampling(landscapes.func1, [-2, 2, -2, 2])
    x2, y2, vals2, bestx2, besty2, best2 = randomSampling(landscapes.func2, [-3, 3, -3, 3])
    fig1, ax1 = plt.subplots()
    print("Function 1:", bestx1, besty1, best1)
    print("Function 2:", bestx2, besty2, best2)
    X, Y = np.meshgrid(np.linspace(-2, 2, 300),
                    np.linspace(-2, 2, 300))
    plot1 = ax1.contour(X, Y, landscapes.func1(X, Y),
                levels=np.linspace(-0.25, 2, 20))


    ax1.scatter(x1, y1, color="red", s=4)
    ax1.clabel(plot1, inline=True, fontsize = 8)
    fig1.colorbar(plot1, ax=ax1, label="Z Value")
    ax1.set_title("Filled Contour Plot with Line Labels")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.scatter(bestx1, besty1, color="blue", marker="*", s=150, label="Best")
    plt.show()

    fig2, ax2 = plt.subplots()

    X2, Y2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    plot2 = ax2.contour(X2, Y2, landscapes.func2(X2, Y2), [-5.8, -5.3, -4.6, -4.0, -3.7, -3.5, -2, -1, 0, 1, 2, 3])
    ax2.clabel(plot2, inline=True, fontsize = 8)
    fig2.colorbar(plot2, ax=ax2, label="Z Value")
    ax2.set_title("Filled Contour Plot with Line Labels")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.scatter(x2, y2, color="red", marker="x", s=4)
    ax2.scatter(bestx2, besty2, color="blue", marker="*", s=150, label="Best")
    plt.show()


starts1 = [(-1.6, -1.4), (-1.4, 1.3), (-0.5, 0),
           (0.5, -1.2), (1.4, 1.2), (1.7, -0.5)]

starts2 = [(-2.4, -2.3), (-2.3, 1.2), (0, -2.4),
           (0.1, 2.3), (2.4, 1.0), (2.3, -1.4)]

functions = [landscapes.func1, landscapes.func2]
bounds = [[-2, 2, -2, 2], [-3, 3, -3, 3]]
starts = [starts1, starts2]
T0s = [[0.05, 1.0, 1.0], [0.1, 3.0, 3.0]]
alpha = [0.999, 0.999, 0.99]
Tmin = 1e-3

s = 0.05
#HillClimb
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

#Annealing
for jj in range(2):
    for ii in range(3):
        f = functions[jj]
        T = T0s[jj][ii]
        a = alpha[ii]
        b = bounds[jj]

        x, y = np.meshgrid(np.linspace(b[0], b[1], 300), np.linspace(b[2], b[3], 300))

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        ax1.contour(x, y, f(x, y), levels=20)

        for i, (x0, y0) in enumerate(starts[jj]):
            path, values, n, reason, bestPoint, bestF, finalT = simAnnealing(f, x0, y0, s, b, T0 = T, alpha = a)


            ax1.plot(path[:, 0], path[:, 1],
                        label=f"Run {i+1}")
            ax1.scatter(x0, y0, marker="o")
            ax1.scatter(*path[-1], marker="x")

            ax2.plot(values, label=f"Run {i+1}, Function {jj}")

            print(f"Function {jj+1}, Schedule {ii+1}, Run {i+1}")
            print(f"Start: ({x0}, {y0})")
            print(f"Final: {path[-1]}")
            print(f"Best: {bestPoint}")
            print(f"Best Value: {bestF:.6f}")
            print(f"Attempts: {n}, Reason: {reason}")
            print(f"Final Temperature: {finalT:.6f}\n")

        ax1.set_title(f"Function {jj+1}: Annealing Schedule {ii+1}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.legend()

        ax2.set_title(f"Function {jj+1}: Annealing Schedule {ii+1}")
        ax2.set_xlabel("Attempted Step")
        ax2.set_ylabel("Current Objective Value")
        ax2.legend()

        fig1.savefig(f"anneal_paths_{jj+1}_{ii+1}.png")
        fig2.savefig(f"anneal_values_{jj+1}_{ii+1}.png")

    plt.show()

# Shallow well 
shallowStarts = [(1.0, 0.5), (1.95, 1.45)]

for jjj in range(2):
    f = functions[jjj]
    b = bounds[jjj]
    x, y = np.meshgrid(np.linspace(b[0], b[1], 300), np.linspace(b[2], b[3], 300))
    for k in range(3):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ax1.contour(x, y, f(x, y), levels=20)


        for i in range(6):
            path, values, n, reason, bestPoint, bestF, finalT = simAnnealing(
                functions[jjj], *shallowStarts[jjj], s, bounds[jjj],
                T0=T0s[jjj][k], alpha=alpha[k]
            )
            ax1.plot(path[:, 0], path[:, 1], label=f"Run {i+1}")
            ax1.scatter(*bestPoint, marker="x")

            ax2.plot(values, label=f"Run {i+1}")

            print(f"Function {jjj+1}, Schedule {k+1}, Run {i+1}")
            print(f"Run {i+1}: Best = {bestF}, Position = {bestPoint}, Steps = {n}")

        ax1.scatter(*shallowStarts[jjj], color="black",
                    marker="*", s=120, label="Start")

        ax1.set_title(f"Function {jjj+1}: Shallow Well, Schedule {k+1}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_aspect("equal")
        ax1.legend()

        ax2.set_title(f"Function {jjj+1}: Shallow Well, Schedule {k+1}")
        ax2.set_xlabel("Attempted Step")
        ax2.set_ylabel("Current Objective Value")
        ax2.legend()

        fig1.savefig(f"shallow_paths_{jjj+1}_{k+1}.png")
        fig2.savefig(f"shallow_values_{jjj+1}_{k+1}.png")

plt.show()
           