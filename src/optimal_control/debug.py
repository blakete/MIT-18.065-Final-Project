import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # load numpy data from `/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data/LQR_x.npy`
    x = np.load('/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data/LQR_x.npy')
    y = np.load('/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data/LQR_y.npy')
    t = np.load('/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data/LQR_time.npy')

    # do this for each segment
    margin = 5.0
    x_min = np.min(x[:,0]) - margin
    x_max = np.max(x[:,0]) + margin

    for i in range(4):
        start_idx = i*101
        end_idx = start_idx + 101
    
        # plt.figure()
        # plt.plot(t[start_idx:end_idx], x[start_idx:end_idx,0], label='x')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(x_min, x_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/offline_path_x_{i}.png')
        # plt.close()

        # plt.figure()
        # plt.plot(t[start_idx:end_idx], x[start_idx:end_idx,1], label='y')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(x_min, x_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/offline_path_y_{i}.png')
        # plt.close()

        # do this for each segment
        margin = 1.0
        t_segment = t[0, start_idx:end_idx]
        x_segment = x[0, start_idx:end_idx]
        y_segment = y[0, start_idx:end_idx]
        print(start_idx, end_idx, x_segment.size)

        x_min = np.min(x_segment) - margin
        x_max = np.max(x_segment) + margin
        y_min = np.min(y_segment) - margin
        y_max = np.max(y_segment) + margin
        
        # plt.figure()
        # plt.plot(t_segment, x_segment, label='x')
        # plt.legend()
        # # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(x_min, x_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/offline_path_x_{i}.png')
        # plt.close()

        # plt.figure()
        # plt.plot(t_segment, y_segment, label='y')
        # plt.legend()
        # # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(y_min, y_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/offline_path_y_{i}.png')
        # plt.close()

        plt.figure()
        plt.plot(x_segment, y_segment, label='x-y')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(f'src/camera_ready/optimal_control/data/offline_x_y_{i}.png')
        plt.close()

        