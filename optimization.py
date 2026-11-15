import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(seed = 22)
import landscapes

def randomSampling(f, bounds, sCount= 1000):
    xP = rng.uniform(bounds[0], bounds[1], sCount)
    yP = rng.uniform(bounds[2], bounds[3], sCount)
    vals = f(xP, yP)
    best = np.argmin(vals)
    return xP, yP, vals, (xP[best], yP[best]), vals[best]


x1, y1, vals1, bestLocs1, best1 = randomSampling(landscapes.func1, [-2, 2, -2, 2])
x2, y2, vals2, bestLocs2, best2 = randomSampling(landscapes.func2, [-3, 3, -3, 3])
fig1, ax1 = plt.subplots()

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
plt.show()








def hillClimb(f, x0, y0, s):
    f0 = f(x0, y0)
    checked = 0
    nextStep = direction()
    while f0 < f(x0 + s * np.cos(nextStep), y0 + s * np.sin(nextStep)):
        nextStep = direction()
        checked+=1
    checked+=1
    return x0 + s * np.cos(nextStep), y0 + s * np.sin(nextStep), checked

def direction():
    return rng.random() * 2 * np.pi

