import numpy as np
import matplotlib.pyplot as plt

def func1(x, y):
    return ((x**2 - 1)**2 + 0.2*x + 4*(y-0.5*x**2)**2)

#Columns : amplitude , x center , y center , x width , y width
FEATURES = np.array ([
[-5.0, -1.85, -1.35, 0.48 , 0.62] ,
[-4.3, 0.15 , -1.80, 0.65 , 0.38] ,
[-6.0, 1.65 , -0.70, 0.38 , 0.55] ,
[-4.8, -1.30, 0.75 , 0.58 , 0.42] ,
[-5.5, 0.15 , 1.75 , 0.40 , 0.52] ,
[-3.9, 1.95 , 1.45 , 0.55 , 0.38] ,
[ 3.5, -0.65, -0.65, 0.42 , 0.48] ,
[ 4.0, 0.70 , 0.45 , 0.45 , 0.38] ,
[ 3.0, -1.95, 1.85 , 0.34 , 0.40] ,
[ 2.8, 1.30 , -2.05, 0.32 , 0.35] ,
])
def func2 (x, y):
    x = np. asarray (x, dtype=float)
    y = np. asarray (y, dtype=float)
    z = 0.06 * (x**2 + y**2)
    for amplitude , cx , cy , sx , sy in FEATURES :
        exponent = ((x - cx) / sx)**2 + ((y - cy) / sy)**2
        z = z + amplitude * np.exp ( -0.5 * exponent )
    return z
if __name__ == "__main__":
    min1x = -1.024
    min1y = 0.524
    min2x = 0.974
    min2y = 0.474
    minima2 = np.array([
        [1.64544, -0.69885],
        [0.14972, 1.73740],
        [-1.84237, -1.34725],
        [-1.28987, 0.74622],
        [0.15812, -1.79770],
        [1.93291, 1.44377]
    ])

    print(f"Left minima expected value: -0.2024 \n Left minima calculated value: {func1(-1.024, 0.524)}")
    print(f"Left minima expected value: 0.1974 \n Left minima calculated value: {func1(0.974, 0.474)}")
    print(f"Left minima expected value: 1.005 \n Left minima calculated value: {func1(0.0501, 0.0013)}")


    x1, y1 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    x2, y2 = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    fig1, ax1 = plt.subplots()
    plot1 = ax1.contour(x1, y1, func1(x1, y1), levels = np.linspace(-0.25, 2, 20))
    ax1.clabel(plot1, inline=True, fontsize = 8)
    fig1.colorbar(plot1, ax=ax1, label="Z Value")
    ax1.set_title("Filled Contour Plot with Line Labels")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.scatter(min1x, min1y, color="red", marker="x", s=80)
    ax1.scatter(min2x, min2y, color="red", marker="x", s=80)
    plt.show()

    fig2, ax2 = plt.subplots()
    plot2 = ax2.contour(x2, y2, func2(x2, y2), levels = [-5.8, -5.3, -4.6, -4.0, -3.7, -3.5, -2, -1, 0, 1, 2, 3])
    ax2.clabel(plot2, inline=True, fontsize = 8)
    fig2.colorbar(plot2, ax=ax2, label="Z Value")
    ax2.set_title("Filled Contour Plot with Line Labels")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.scatter(minima2[:,0], minima2[:,1],  color="red", marker="x", s=80)
    plt.show()

    for x, y in minima2:
        print(f"({x}, {y}): {func2(x, y)}")


