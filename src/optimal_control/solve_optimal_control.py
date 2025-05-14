# 18.0651 Final Project
# Optimal Control LQR and Min Time Implementations
import os
import time
from typing import List, Callable

import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L = 1.0  # length of car
N = 100  # discrete steps per segment
SPEED_LIMIT = 0.05
MAX_SPEED_COMMAND = SPEED_LIMIT  # Maximum acceleration per step
MAX_THETA_COMMAND = 10 * (ca.pi / 180)
THETA_DOT_LIMIT = MAX_THETA_COMMAND

# Scaled to match RL formulation
xy_state_limit_margin = 0.5
X_LIMIT_LOWER = 0.0 - xy_state_limit_margin
X_LIMIT_UPPER = 1.0 + xy_state_limit_margin
Y_LIMIT_LOWER = 0.0 - xy_state_limit_margin
Y_LIMIT_UPPER = 1.0 + xy_state_limit_margin

WAYPOINT_CAPTURE_RADIUS = 0.1


def solve_LQR_oc(x0: List[float], xf: List[float]):
    # initialize problem
    opti = ca.Opti()

    # decision variables
    x = opti.variable(N+1, 3)
    u = opti.variable(N, 2)
    # T = 500
    T = opti.variable()
    dt = T / N # time step (symbolic)
    # t vector will be constructed after solving using optimized T

    ## COST
    J = 0
    Q = [50, 50]  # state weights
    R = [15, 15]  # control weights
    H = [1, 1]  # terminal state weights
    for k in range(N+1):
        for i in range(2):  # state LQR
            J += 1/2 * dt * Q[i] * (x[k, i] - xf[i]) ** 2
        if k < N:
            J += 1/2 * dt * (R[0] * u[k, 0] ** 2 + R[1] * u[k, 1] ** 2)  # control LQR
    # terminal condition
    for i in range(2):
        J += 1/2 * H[i] * (x[N, i] - xf[i]) ** 2
    # # add a small cost on time
    # w_time = 1e-2
    # J += w_time*T
    opti.minimize(J)

    ## BOUNDARY CONDITIONS
    # initial condition
    x0_np = np.reshape(np.array(x0), (1, 3))
    opti.subject_to(x[0, :] == x0_np)
    # terminal "capture" constraint: end within R of the waypoint (ignores heading)
    opti.subject_to(((x[N,0] - xf[0])**2 + (x[N,1] - xf[1])**2) <= WAYPOINT_CAPTURE_RADIUS**2)

    # ## TIME BOUNDS
    opti.subject_to(T > 1e-3)   # time must be positive
    opti.subject_to(T < 800)   # time must be bounded

    ## CONSTRAINTS
    # dynamics constraints
    def f(xk, uk0, uk1):
        """
        xk: (x,y,theta) at time step k
        uk0: u speed at time k
        uk1: u steering angle at time k
        """
        dx = ca.vertcat(
            uk0 * ca.cos(xk[2]),
            uk0 * ca.sin(xk[2]),
            (uk0/L) * ca.tan(uk1)
        )
        dx = ca.reshape(dx, (1, 3))
        return dx

    # Runge-Kutta Integration
    for k in range(N):
        k1 = f(x[k, :], u[k, 0], u[k, 1])
        k2 = f(x[k, :] + (dt/2) * k1, u[k, 0], u[k, 1])
        k3 = f(x[k, :] + (dt/2) * k2, u[k, 0], u[k, 1])
        k4 = f(x[k, :] + dt * k3, u[k, 0], u[k, 1])
        x_next = x[k, :] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(x[k + 1, :] == x_next)

    # --- Control Constraints ---
    # Speed Command Limits
    # Add constraint to enforce the physical speed limit
    opti.subject_to(u[:,0] <= SPEED_LIMIT)
    # Add constraint to enforce the physical speed limit
    opti.subject_to(u[:, 0].fabs() <= MAX_SPEED_COMMAND)
    opti.subject_to(u[:, 0] > 0)  # forward motion only
    # steering-angle (wheel) limits so that tan() never blows up
    limit = ca.pi/2 - ca.pi/100
    opti.subject_to(u[:, 1] >= -limit)
    opti.subject_to(u[:, 1] <=  limit)
    # # Heading Command Limits
    # du = u[1:,1] - u[:-1,1]
    # opti.subject_to(ca.fabs(du) <= MAX_THETA_COMMAND * dt)

    # --- State Constraints ---
    # simple box bounds on x and y for every time step
    opti.subject_to(x[:, 0] <= X_LIMIT_UPPER)
    opti.subject_to(x[:, 0] >= X_LIMIT_LOWER)
    opti.subject_to(x[:, 1] <= Y_LIMIT_UPPER)
    opti.subject_to(x[:, 1] >= Y_LIMIT_LOWER)
    # Heading constraint
    dx_theta = x[1:, 2] - x[:-1, 2]
    opti.subject_to(ca.fabs(dx_theta) <= THETA_DOT_LIMIT * dt)
    # Proper speed constraint (Euclidean distance per timestep)
    speed = ca.sqrt((x[1:, 0] - x[:-1, 0])**2 + (x[1:, 1] - x[:-1, 1])**2)
    opti.subject_to(speed <= SPEED_LIMIT * dt)

    ## --- Initial Guesses ---
    # Initial guess for T
    dist = np.linalg.norm(np.array(x0[:2]) - np.array(xf[:2]))
    opti.set_initial(T, ((dist - WAYPOINT_CAPTURE_RADIUS)/MAX_SPEED_COMMAND) + 0.1)
    # Initial guess for controls
    u_init = np.zeros((N, 2))
    u_init[:, 0] = MAX_SPEED_COMMAND * 0.8
    u_init[:, 1] = 0.0
    opti.set_initial(u, u_init)
    # Initial guess for states with a reasonable final heading
    # compute a final heading as the angle from x0 to xf in the xy-plane
    theta_guess = np.arctan2(xf[1] - x0[1], xf[0] - x0[0])
    x_init = np.zeros((N+1, 3))
    for k in range(N+1):
        alpha = k / N
        # linear interpolate position
        x_init[k, 0] = (1 - alpha) * x0[0] + alpha * xf[0]
        x_init[k, 1] = (1 - alpha) * x0[1] + alpha * xf[1]
        # interpolate heading from x0[2] to theta_guess
        x_init[k, 2] = (1 - alpha) * x0[2] + alpha * theta_guess

    opti.set_initial(x, x_init)

    ## SOLVING
    TOL = 1e-20
    opti.solver(
        "ipopt",
        # Casadi flags
        {"print_time": False},
        # Ipopt flags
        {
            "tol": TOL,
            "constr_viol_tol": TOL,
            "acceptable_tol": TOL,
            "acceptable_dual_inf_tol": TOL,
            # ---- Silence all printing ----
            "print_level": 3,
            "file_print_level": 1,
            "sb": "yes",
            "timing_statistics": "no",
            # "output_file": "/dev/null",
        },
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug

    ## EXTRACT SOLUTION
    x_opt = sol.value(x)
    u_opt = sol.value(u)
    T_opt = sol.value(T)
    t_opt = np.linspace(0, T_opt, N+1)
    
    return t_opt, x_opt, u_opt


def solve_min_time_oc(x0: List[float], xf: List[float]):
    # initialize problem
    opti = ca.Opti()

    # decision variables
    x = opti.variable(N+1, 3)
    u = opti.variable(N, 2)
    T = opti.variable()
        
    dt = T / N # time steps 

    ## COST
    # opti.minimize(T + 1e-2 * ca.sumsqr(u[:,1])) # added a small cost on steering and now car likes to go straight!
    opti.minimize(T) # minimize time only
    # ## TIME BOUNDS
    opti.subject_to(T > 1e-3)   # time must be positive
    opti.subject_to(T < 1000)   # time must be bounded
    # # --- Physical lower bound on T so we can actually get within the capture radius ---
    # # compute straight-line distance in xy:
    # # (we already did this for the initial guess; repeat it symbolically here)
    # delta_xy = xf[:2] - x0[:2]
    # dist = ca.norm_2(delta_xy)
    # min_phys = (dist - WAYPOINT_CAPTURE_RADIUS) / MAX_SPEED_COMMAND
    # opti.subject_to(T >= ca.fmax(min_phys, 0))

    ## BOUNDARY CONDITIONS
    # initial condition
    x0_np = np.reshape(np.array(x0), (1, 3))
    opti.subject_to(x[0,:] == x0_np) 

    # terminal "capture" constraint: end within R of the waypoint (ignores heading)
    opti.subject_to((x[N,0] - xf[0])**2 + (x[N,1] - xf[1])**2 <= WAYPOINT_CAPTURE_RADIUS**2)

    ## CONSTRAINTS
    # dynamics constraints 
    def f(xk, uk0, uk1):
        """
        xk: (x,y,theta) at time step k
        uk0: u speed at time k
        uk1: u steering angle at time k
        """
        dx = ca.vertcat(
            uk0 * ca.cos(xk[2]),
            uk0 * ca.sin(xk[2]),
            (uk0/L) * ca.tan(uk1)
        )
        dx = ca.reshape(dx, (1,3))
        return dx
        
    for k in range(N):
        # Runge-Kutta Integration
        k1 = f(x[k, :], u[k,0], u[k,1])
        k2 = f(x[k, :] + (dt/2) * k1, u[k,0], u[k,1])
        k3 = f(x[k, :] + (dt/2) * k2, u[k,0], u[k,1])
        k4 = f(x[k, :] + dt * k3, u[k,0], u[k,1])
        # using diff order derivs for forward prop
        x_next = x[k,:] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(x[k + 1, :] == x_next)

    # --- Control Constraints ---
    # Speed Command Limits
    # Add constraint to enforce the physical speed limit
    opti.subject_to(u[:, 0].fabs() <= MAX_SPEED_COMMAND)
    opti.subject_to(u[:, 0] > 0)  # forward motion only
    # Add constraint to enforce the physical speed limit
    opti.subject_to(u[:,0] <= SPEED_LIMIT)
    # steering-angle (wheel) limits so that tan() never blows up
    limit = ca.pi/2 - ca.pi/100   # stay away from the tan() singularity
    opti.subject_to(u[:,1] >= -limit)
    opti.subject_to(u[:,1] <= limit)
    # # Heading Command Limits
    # du = u[1:,1] - u[:-1,1]
    # opti.subject_to(ca.fabs(du) <= MAX_THETA_COMMAND * dt)

    # --- State Constraints ---
    # simple box bounds on x and y for every time step
    opti.subject_to(x[:, 0] <= X_LIMIT_UPPER)
    opti.subject_to(x[:, 0] >= X_LIMIT_LOWER)
    opti.subject_to(x[:, 1] <= Y_LIMIT_UPPER)
    opti.subject_to(x[:, 1] >= Y_LIMIT_LOWER)
    # Heading constraint
    dx_theta = x[1:, 2] - x[:-1, 2]
    opti.subject_to(ca.fabs(dx_theta) <= THETA_DOT_LIMIT * dt)
    # Proper speed constraint (Euclidean distance per timestep)
    speed = ca.sqrt((x[1:, 0] - x[:-1, 0])**2 + (x[1:, 1] - x[:-1, 1])**2)
    opti.subject_to(speed <= SPEED_LIMIT * dt)

    # --- Initial Guesses ---
    # Initial guess for T
    dist = np.linalg.norm(np.array(x0[:2]) - np.array(xf[:2]))
    opti.set_initial(T, ((dist - WAYPOINT_CAPTURE_RADIUS)/MAX_SPEED_COMMAND) + 0.1)

    # --- Initial Guesses for Controls ---
    u_init = np.zeros((N, 2))
    u_init[:, 0] = MAX_SPEED_COMMAND * 0.8  # start with 80% of max speed
    u_init[:, 1] = 0.0  # no steering initially
    opti.set_initial(u, u_init)

    # Initial guess for states with a reasonable final heading
    # compute a final heading as the angle from x0 → xf in the xy-plane
    theta_guess = np.arctan2(xf[1] - x0[1], xf[0] - x0[0])
    x_init = np.zeros((N+1, 3))
    for k in range(N+1):
        alpha = k / N
        # linear interpolate position
        x_init[k, 0] = (1 - alpha) * x0[0] + alpha * xf[0]
        x_init[k, 1] = (1 - alpha) * x0[1] + alpha * xf[1]
        # interpolate heading from x0[2] to theta_guess
        x_init[k, 2] = (1 - alpha) * x0[2] + alpha * theta_guess

    opti.set_initial(x, x_init)

    ## SOLVING
    TOL = 1e-20
    opti.solver(
        "ipopt",
        # Casadi flags
        {"print_time": False},
        # Ipopt flags
        {
            "tol": TOL,
            "constr_viol_tol": TOL,
            "acceptable_tol": TOL,
            "acceptable_dual_inf_tol": TOL,
            # ---- Silence all printing ----
            "print_level": 3,
            "file_print_level": 1,
            "sb": "yes",
            "timing_statistics": "no",
            # "output_file": "/dev/null",
        },
    )
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug


    ## EXTRACT SOLUTION
    x = sol.value(x)
    u = sol.value(u)
    T = sol.value(T)

    ## PLOTTING
    t = np.linspace(0, T, N + 1)

    return t, x, u


def solve_ocp_for_race_waypoints(path: List[List[float]], func: Callable):
    """
    runs manual phase solution strategy for race car
    """
    t_x = []
    x_x = []
    x_y = []
    x_a = []
    t_u = []
    u_s = []
    u_h = []
    start_t = 0
    path_start = path[0]
    waypoint_segment_times = []
    
    for p in range(len(path) - 1):
        start_t_waypoint_segment = time.time()
        t, x, u = func(path_start, path[p+1])
        end_t_waypoint_segment = time.time()
        print(f"Waypoint segment {p} solution time: {end_t_waypoint_segment - start_t_waypoint_segment} seconds")

        # # do this for each segment
        # margin = 5.0
        # x_min = np.min(x[:,0]) - margin
        # x_max = np.max(x[:,0]) + margin
        
        # plt.figure()
        # plt.plot(t, x[:,0], label='x')
        # plt.legend()
        # # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(x_min, x_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/path_x_{p}.png')
        # plt.close()

        # plt.figure()
        # plt.plot(t, x[:,1], label='y')
        # plt.legend()
        # # plt.gca().set_aspect('equal', adjustable='box')
        # plt.ylim(x_min, x_max)
        # plt.savefig(f'src/camera_ready/optimal_control/data/path_y_{p}.png')
        # plt.close()

        waypoint_segment_times.append(end_t_waypoint_segment - start_t_waypoint_segment)
        
        # Add trajectory segment data - skip the first point on all but the first segment
        # to avoid duplicating points at segment boundaries
        start_idx = 0 if p == 0 else 1
        for i in range(start_idx, len(t)):
            t_x.append(t[i] + start_t)
            x_x.append(x[i,0])
            x_y.append(x[i,1])
            x_a.append(x[i,2])
            if i < len(t) - 1:
                # record the exact time for each control step
                t_u.append(t[i] + start_t)
                u_s.append(u[i,0])
                u_h.append(u[i,1])
        
        # # Update start time for next segment
        # # Ensure the time difference is at least a small non-zero value to prevent div by zero
        start_t = t_x[-1]
        # if p < len(path) - 2:  # If not the last segment
        #     # Add a tiny time increment to avoid zero dt between segments
        #     # This is much smaller than before but still prevents division by zero
        #     min_dt = 0.001  # 1ms minimum time step
        #     start_t = max(start_t + min_dt, start_t)
        
        # # Calculate the time difference between the last point of the current segment and the first point of the next segment
        # if p < len(path) - 2:  # Only if not the last segment
        #     last_point_time = t_x[-1]
        #     next_segment_start_time = t_x[0]
        #     time_diff = next_segment_start_time - last_point_time
        #     if time_diff < 0.001:
        #         print("Warning: Time difference between segments is too small")
        path_start = x[-1, :] # start next path at the end of the previous path

    t_x = np.reshape(np.array(t_x), (1,len(t_x)))
    x_x = np.reshape(np.array(x_x), (1,len(x_x)))
    x_y = np.reshape(np.array(x_y), (1,len(x_y)))
    x_a = np.reshape(np.array(x_a), (1,len(x_a)))
    t_u = np.reshape(np.array(t_u), (1,len(t_u)))
    u_s = np.reshape(np.array(u_s), (1,len(u_s)))
    u_h = np.reshape(np.array(u_h), (1,len(u_h)))

    return t_x, x_x, x_y, x_a, t_u, u_s, u_h


def save_data(ti, x_x, x_y, x_a, t_u, u_s, u_h, identifier, output_data_dir):
    os.makedirs(output_data_dir, exist_ok=True)
    np.save(os.path.join(output_data_dir, f'{identifier}_time.npy'), ti)
    np.save(os.path.join(output_data_dir, f'{identifier}_x.npy'), x_x)
    np.save(os.path.join(output_data_dir, f'{identifier}_y.npy'), x_y)
    np.save(os.path.join(output_data_dir, f'{identifier}_theta.npy'), x_a)
    # save control timestamps
    np.save(os.path.join(output_data_dir, f'{identifier}_time_u.npy'), t_u)
    np.save(os.path.join(output_data_dir, f'{identifier}_speed.npy'), u_s)
    np.save(os.path.join(output_data_dir, f'{identifier}_heading.npy'), u_h)


def solve_optimal_control_problems():
    ### SOLVE OCPs FOR RACE TRACK
    output_data_dir = '/Users/blake/repos/18.065-Final-Project/src/camera_ready/optimal_control/data'

    # Load Race Track Waypoints
    race_course_waypoints = "/Users/blake/repos/18.065-Final-Project/src/camera_ready/comparison_and_analysis/race_track_waypoints/square_waypoints.csv"
    race_track_waypoints = pd.read_csv(race_course_waypoints).values.tolist()
    # NOTE: angles are only used for initial guesses
    angles = [0, np.nan, np.nan, np.nan, np.nan]
    for i in range(len(race_track_waypoints)):
        race_track_waypoints[i].append(angles[i])
    race_track_waypoints = np.array(race_track_waypoints)

    # Solve LQR Optimal Control Problem
    print("\nSolving LQR Optimal Control Problem...")
    start_time = time.time()
    ti_L, x_x_L, x_y_L, x_a_L, t_u_L, u_s_L, u_h_L = solve_ocp_for_race_waypoints(race_track_waypoints, solve_LQR_oc)
    LQR_solution_time = time.time() - start_time
    print(f"LQR solution time: {LQR_solution_time} seconds")
    # Save Results
    save_data(ti_L, x_x_L, x_y_L, x_a_L, t_u_L, u_s_L, u_h_L, 'LQR', output_data_dir)
    
    # # Solve Min Time Optimal Control Problem
    # print("\nSolving Min Time Optimal Control Problem...")
    # start_time = time.time()
    # ti_T, x_x_T, x_y_T, x_a_T, t_u_T, u_s_T, u_h_T = solve_ocp_for_race_waypoints(race_track_waypoints, solve_min_time_oc)
    # MT_solution_time = time.time() - start_time
    # print(f"MT solution time: {MT_solution_time} seconds\n")
    # # Save Results
    # save_data(ti_T, x_x_T, x_y_T, x_a_T, t_u_T, u_s_T, u_h_T, 'MT', output_data_dir)

    print("Complete!")


if __name__ == "__main__":
    solve_optimal_control_problems()
    