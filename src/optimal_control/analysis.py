"""
# Intent of This File
-

# Checklist
- [ ] Comparing performance & visualization of results
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from matplotlib import animation
import matplotlib.transforms as transforms


## VISUALIZATION

def plotStates(time, states):
    plt.figure()
    plt.plot(time, states[0], '.-')
    plt.plot(time, states[1], '.-')
    plt.plot(time, states[2], '.-')
    # plt.plot(time, states[3], '.-')
    # plt.plot(time, states[4], '.-')
    # plt.plot(time, states[5], '.-')
    plt.legend(('x', 'y', 'theta'))
    plt.title('States')
    plt.xlabel('Time (s)')


def plotPath(x, y, path):
    plt.figure()
    plt.plot(x, y, '.-')
    for p in path:
        # print(p)
        plt.plot(p[0], p[1], 'r.')
    plt.title('Vehicle Position')
    plt.xlabel('x')
    plt.ylabel('y')


def plotControl(time, controls):
    plt.figure()
    plt.plot(time, controls[0])
    plt.plot(time, controls[1])
    plt.legend(('speed', 'heading'))
    plt.title('Control')
    plt.xlabel('Time (s)')


def animateCar(t, x, y, theta):  # best plotter so far, from ChatGPT
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    x_min = min(x) - 1
    x_max = max(x) + 1
    y_min = min(y) - 1
    y_max = max(y) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Define car size
    car_length = 1.0  # along the heading (long direction)
    car_width = 0.5  # across the heading (short direction)

    # Create car rectangle, centered at (0,0)
    patch = Rectangle((-car_length / 2, -car_width / 2), car_length, car_width, fc='b')
    ax.add_patch(patch)

    # Create a circle to mark the front
    front_marker = Circle((car_length / 2, 0), 0.1, fc='r')  # circle at front center
    ax.add_patch(front_marker)

    for i in range(len(t)):
        # Transformation for rotation and translation
        trans = transforms.Affine2D().rotate(theta[i]).translate(x[i], y[i]) + ax.transData

        patch.set_transform(trans)
        front_marker.set_transform(trans)

        plt.plot(x[i], y[i], 'k.', markersize=2)  # trail of points
        plt.pause(0.03)

    plt.show()


def animateMultipleCars(ts, xs, ys, thetas, path):
    """
    Input: 
    - ts        --> time steps, ROUNDED to 0.01 increments
    - xs        --> x world positions 
    - ys        --> y world positions
    - thetas    --> theta world positions
    * shape of all --> (numCars, numTimeSteps)

    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    margin = 0.5
    x_min = np.min(xs) - margin
    x_max = np.max(xs) + margin
    y_min = np.min(ys) - margin
    y_max = np.max(ys) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for px, py in path:
        circ = Circle((px, py), radius=0.1, facecolor='none', edgecolor='green', linewidth=1.5, alpha=0.8)
        ax.add_patch(circ)

    numCar = np.shape(ts)[0]

    # Define car size
    car_length = 0.1  # along the heading (long direction)
    car_width = 0.05  # across the heading (short direction)

    # Create car rectangle, centered at (0,0)
    patches = []
    markers = []
    colors = ['k', 'r']
    for car in range(numCar):
        patch = Rectangle((-car_length / 2, -car_width / 2), car_length, car_width, fc='b')
        ax.add_patch(patch)
        patches.append(patch)

        # Create a circle to mark the front
        front_marker = Circle((car_length / 2, 0), 0.01, fc=colors[car])  # circle at front center
        ax.add_patch(front_marker)
        markers.append(front_marker)

    # print(ts.shape, ts[:, -1], max(ts[:,-1]), (int(max(ts[:,-1])) + 0.05)/0.01)
    # steps = np.linspace(0, max(ts[:,-1]), int((int(max(ts[:,-1])))/0.01))
    steps = np.arange(0, max(ts[:, -1]), 0.01)
    plt.pause(1.00)
    plt.title('minTime = bk (59s), LQR = r (40s), time: \n0.0')
    count = 0
    for t in steps:
        t = round(t, 2)
        # Transformation for rotation and translation
        for car in range(numCar):
            if t in ts[car, :]:
                i = np.where(ts[car, :] == t)
                if len(i[0]) > 1:
                    i = i[0][0]

                trans = transforms.Affine2D().rotate(thetas[car, i]).translate(xs[car, i], ys[car, i]) + ax.transData

                patches[car].set_transform(trans)
                markers[car].set_transform(trans)

                plt.plot(xs[car, i], ys[car, i], colors[car] + '.', markersize=2)  # trail of points
                if count % 10 == 0:
                    plt.title('minTime = bk (59s), LQR = r (40s), time: \n' + str(t))
        count += 1
        plt.pause(0.001)

    plt.show()

    # for car in range(numCar):
    #     plotStates(ts[car,:], [xs[car,:], ys[car,:], thetas[car,:]])
    # plt.show()


from rich.progress import track
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms
import imageio


def animateMultipleCars_to_mp4(ts, xs, ys, thetas, path, dt=2.0, fps=5, output_path='animation.mp4'):
    """
    Generates all frames up-front, shows a rich-text progress bar,
    plots each car's trail up to the current frame,
    then saves to MP4 at specified FPS.
    Uses print_to_buffer() to capture frames in backends like QTAgg.
    
    Inputs:
    - ts, xs, ys, thetas : arrays shape (n_cars, n_steps_i)
    - path : list of (px,py) waypoints
    - dt : time-step for interpolation (s)
    - fps : frames per second in the output video
    - output_pat : filename for the saved MP4
    """
    n_cars = ts.shape[0]
    # 1) build uniform time vector
    t_max = np.max(ts[:, -1])
    uniform_t = np.arange(0.0, t_max + dt / 2, dt).round(2)
    n_frames = len(uniform_t)

    # 2) unwrap thetas for smooth interpolation
    thetas_un = np.unwrap(thetas, axis=1)

    # 3) precompute interpolated trajectories
    xs_i = np.array([np.interp(uniform_t, ts[i], xs[i], right=xs[i, -1]) for i in range(n_cars)])
    ys_i = np.array([np.interp(uniform_t, ts[i], ys[i], right=ys[i, -1]) for i in range(n_cars)])
    ths_i = np.array([np.interp(uniform_t, ts[i], thetas_un[i], right=thetas_un[i, -1]) for i in range(n_cars)])

    # 4) set up figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    margin = 0.5
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)

    # draw waypoints
    for px, py in path:
        ax.add_patch(Circle((px, py), 0.1, facecolor='none',
                            edgecolor='green', lw=1.5, alpha=0.8))

    # initialize car patches
    car_length, car_width = 0.1, 0.05
    patches, markers = [], []
    colors = ['k', 'r'] * ((n_cars + 1) // 2)
    for i in range(n_cars):
        rect = Rectangle((-car_length / 2, -car_width / 2), car_length, car_width, fc='blue', alpha=0.7)
        mk = Circle((car_length / 2, 0), 0.01, fc=colors[i])
        ax.add_patch(rect)
        ax.add_patch(mk)
        patches.append(rect)
        markers.append(mk)

    title = ax.text(0.5, 1.02, '', transform=ax.transAxes, ha='center', va='bottom')
    frames = []

    # 5) generate all frames with progress bar, plotting trails
    for idx, t in track(enumerate(uniform_t), total=n_frames, description="Rendering frames"):
        for i in range(n_cars):
            # update car transform
            trans = (transforms.Affine2D()
                     .rotate(ths_i[i, idx])
                     .translate(xs_i[i, idx], ys_i[i, idx])
                     + ax.transData)
            patches[i].set_transform(trans)
            markers[i].set_transform(trans)
            # plot trail point
            ax.plot(xs_i[i, :idx + 1], ys_i[i, :idx + 1], colors[i] + '.', markersize=2)

        title.set_text(f"time = {t:.2f} s")

        fig.canvas.draw()
        # capture using print_to_buffer
        buf, size = fig.canvas.print_to_buffer()
        w, h = size
        img = np.frombuffer(buf, dtype='uint8').reshape(h, w, 4)[..., :3]
        frames.append(img)

    plt.close(fig)

    # 6) write MP4
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved animation to {output_path}")

    return output_path

# Example usage:
# animateMultipleCars_to_mp4(ts, xs, ys, thetas, path, dt=0.01, fps=5, output_path='out.mp4')


## PERFORMANCE
# distance from waypoint
# error of point to final state
# time taken
# energy used
# compared to direct shot line

# convergence rates
