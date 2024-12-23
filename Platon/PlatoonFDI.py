
from time import time
import numpy as np
import matplotlib.pyplot as plt
import random


class DynamicBicycleModel:
    def __init__(self, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.psi = psi  # Yaw angle
        self.v_x = vx  # Longitudinal velocity
        self.v_y = vy  # Lateral velocity
        self.r = r  # Yaw rate

        # Vehicle-specific parameters
        self.m = 1500  # Vehicle mass kg
        self.L_f = 1.2  # Distance from CG to front axle (m)
        self.L_r = 1.6  # Distance from CG to rear axle (m)
        self.I_z = 2250  # Yaw moment of inertia (kg*m^2)
        self.C_f = 19000  # Cornering stiffness front (N/rad)
        self.C_r = 20000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)

    def update(self, a, delta):
        """
        Update the vehicle's state based on the bicycle model dynamics
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])

        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt / 2 * k1, a, delta)
        k3 = self.f(xx + self.dt / 2 * k2, a, delta)
        k4 = self.f(xx + self.dt * k3, a, delta)

        xx = xx + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        self.x = xx[0]  # Global x position
        self.y = xx[1]  # Global y position
        self.psi = xx[2]  # Yaw angle
        self.v_x = xx[3]  # Longitudinal velocity
        self.v_y = xx[4]  # Lateral velocity
        self.r = xx[5]  # Yaw rate

    def f(self, xx, a, delta):
        """
        calculating the states' derivatives using Runge-Kutta Algorithm
        """
        x = xx[0]  # Global x position
        y = xx[1]  # Global y position
        psi = xx[2]  # Yaw angle
        v_x = xx[3]  # Longitudinal velocity
        v_y = xx[4]  # Lateral velocity
        r = xx[5]  # Yaw rate

        # Calculate slip angles
        alpha_f = delta - np.arctan2((v_y + self.L_f * r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r * r), v_x)

        # Calculate lateral forces
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # State variables
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)  # Global x position
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)  # Global y position
        psi_dot = r  # Yaw angle
        v_x_dot = a - (F_yf * np.sin(delta)) / self.m + v_y * r  # Longitudinal velocity
        v_y_dot = (F_yf * np.cos(delta) + F_yr) / self.m - v_x * r  # Lateral velocity
        r_dot = (self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr) / self.I_z  # Yaw rate

        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }



class LeaderFollowerSimulation:
    def __init__(self, num_followers):
        self.leader = DynamicBicycleModel( id=0, x=100, y=10, vx=6.0)
        # Initialize followers with slightly different initial velocities
        self.followers = [
            DynamicBicycleModel( 
                id=i + 1, 
                x=100 - 10 * (i + 1), 
                y=10, 
                vx=6.0   # Add random initial velocity offset
            ) for i in range(num_followers)
        ]
        self.num_followers = num_followers
        self.desired_gap = 10  # Desired gap between vehicles (m)
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20  # Width of the road (meters)

    def run_simulation(self):
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]
        min_distances = []
        time_points = np.arange(0, self.time_steps) * self.dt

        for t in range(self.time_steps):
            # Update leader's state
            v_target = 6.0
            k_p = 1
            a_l = k_p * (v_target - self.leader.v_x)
            self.leader.update(0, 0)

            # Save leader's position and velocity
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')

            # Update each follower with modified control gains
            for i, follower in enumerate(self.followers):
                # Modify the control gain for each follower to show different convergence rates
                k_p_follower = 1 
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = k_p_follower * distance_to_leader
                # if 30 <t< 35 and i == 2:
                #     # Modify the velocity drastically to simulate an attack
                #     follower.v_x += random.uniform(5, 10)  # Add an unrealistic jump in velocity
                #     print(f"Cyber attack introduced to vehicle {follower.id} at step {t}")

                follower.update(0, 0)

                x_history[i + 1].append(follower.x)
                y_history[i + 1].append(follower.y)
                v_history[i + 1].append(follower.v_x)

                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                 (follower.y - self.followers[i-1].y)**2)
                min_dist_timestep = min(min_dist_timestep, dist)

            min_distances.append(min_dist_timestep)

        # Plot 1: Trajectory snapshots [same as before]
        plt.figure(figsize=(15, 8))
        plt.suptitle('Vehicle Platooning Trajectory Snapshots', fontsize=14)
        t_samples = [int(self.time_steps * 0.01), int(self.time_steps * 0.2), int(self.time_steps * 0.4),
                     int(self.time_steps * 0.6), int(self.time_steps * 0.8), int(self.time_steps * 0.99)]
        
        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers + 1):
                plt.plot(x_history[i][t], y_history[i][t], '.k' if i > 0 else '.r', markersize=10 if i == 0 else 5)

            plt.plot([0, self.leader.x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, self.leader.x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t * self.dt:.2f} sec')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

        # Plot 2: Velocity consensus with zoomed y-axis
        plt.figure(figsize=(10, 6))
        # Use a different colormap with more distinct colors
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_followers + 1))
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=2.5)  # Increased line width
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity Consensus Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(5, 15)
        plt.tight_layout()
        plt.show()

        # Plot 3: Minimum distances [same as before]
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.title('Minimum Inter-Vehicle Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(5, 15)
        plt.tight_layout()
        plt.show()

# Run the simulation
num_followers = 10
simulation = LeaderFollowerSimulation(num_followers)
simulation.run_simulation()
