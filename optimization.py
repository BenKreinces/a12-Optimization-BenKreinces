import numpy as np
import matplotlib.pyplot as plt
import landscapes

rng = np.random.default_rng(seed = 22)

def randomSampling(f, bounds, sCount= 1000):
    if not isinstance(sCount, (int, np.integer)) or sCount <= 0:
        raise ValueError('sCount must be a positive integer')
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('bounds must be [xmin, xmax, ymin, ymax]')
    xP = rng.uniform(bounds[0], bounds[1], sCount)
    yP = rng.uniform(bounds[2], bounds[3], sCount)
    vals = f(xP, yP)
    best = np.argmin(vals)
    return xP, yP, vals, xP[best], yP[best], vals[best]

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



def hillClimb(f, x0, y0, s, bounds, Nmax=10000, W=100, epsilon=1e-4):
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('bounds must be [xmin, xmax, ymin, ymax]')
    if not (bounds[0] <= x0 <= bounds[1] and bounds[2] <= y0 <= bounds[3]):
        raise ValueError('starting point is outside bounds')
    if s <= 0 or Nmax <= 0 or W <= 0 or W > Nmax or epsilon < 0:
        raise ValueError('require s>0, Nmax>=W>0, epsilon>=0')
    f0 = f(x0, y0)
    path = [(x0, y0)]
    values = [f0]
    reason = 'max steps'
    for i in range(Nmax):
        nextStep = direction()
        newX = x0 + s * np.cos(nextStep)
        newY = y0 + s * np.sin(nextStep)
        if bounds[0] <= newX <= bounds[1] and bounds[2] <= newY <= bounds[3]:
            newF = f(newX, newY)
            if newF < f0:
                x0, y0 = newX, newY
                f0 = newF
        path.append((x0, y0))
        values.append(f0)

        if i + 1 >= W:
            if values[-W-1] - values[-1] <= epsilon: #w steps ago compared to now
                reason = 'little improvement'
                break
    return np.array(path), np.array(values), i + 1, reason

def direction():
    return rng.random() * 2 * np.pi

def simAnnealing(f, x0, y0, s, bounds, Nmax=10000, W=100, epsilon=1e-4, T0 = 0.5, alpha = 0.995):
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('bounds must be [xmin, xmax, ymin, ymax]')
    if not (bounds[0] <= x0 <= bounds[1] and bounds[2] <= y0 <= bounds[3]):
        raise ValueError('starting point is outside bounds')
    if s <= 0 or Nmax <= 0 or W <= 0 or W > Nmax or epsilon < 0:
        raise ValueError('require s>0, Nmax>=W>0, epsilon>=0')
    if T0 <= 0 or not (0 < alpha < 1):
        raise ValueError("Invalid temperature parameters")
    f0 = f(x0, y0)
    path = [(x0, y0)]
    values = [f0]
    bestX, bestY, bestF = x0, y0, f0
    bestValues = [bestF]
    T = T0
    reason = 'max steps'
    for i in range(Nmax):
        nextStep = direction()
        newX = x0 + s * np.cos(nextStep)
        newY = y0 + s * np.sin(nextStep)
        if bounds[0] <= newX <= bounds[1] and bounds[2] <= newY <= bounds[3]:
            newF = f(newX, newY)
            if newF <= f0 or rng.random() < np.exp(-(newF - f0) / T):
                x0, y0 = newX, newY
                f0 = newF
                if f0 < bestF:
                    bestX, bestY, bestF = x0, y0, f0
        path.append((x0, y0))
        values.append(f0)
        bestValues.append(bestF)
        T *= alpha
        if i + 1 >= W:
            if T<=1e-3 and bestValues[-W-1] - bestValues[-1] <= epsilon: #w steps ago compared to now
                reason = 'little improvement'
                break
    return np.array(path), np.array(values), i + 1, reason, (bestX, bestY), bestF, T



