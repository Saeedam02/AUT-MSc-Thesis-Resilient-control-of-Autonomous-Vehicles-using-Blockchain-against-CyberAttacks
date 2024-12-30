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
        v_y_dot = (F_yf * np.cos (delta) + F_yr) / self.m - v_x * r  # Lateral velocity
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
        self.leader = DynamicBicycleModel(id=0, x=100, y=10, vx=6.0)
        self.followers = [
            DynamicBicycleModel(
                id=i + 1,
                x=100 - 10 * (i + 1),
                y=10,
                vx=6.0
            ) for i in range(num_followers)
        ]
        self.num_followers = num_followers
        self.desired_gap = 10  # Desired gap between vehicles (m)
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20  # Width of the road (meters)
        self.ttc_history = []  # Track minimum TTC over time

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        
        # Leader-to-first-follower TTC
        leader = self.leader
        first_follower = self.followers[0]
        relative_velocity = first_follower.v_x - leader.v_x
        relative_position = leader.x - first_follower.x - self.desired_gap

        if relative_position > 0 and relative_velocity > 0:
            ttc = relative_position / relative_velocity
        else:
            ttc = float('inf')  # No collision risk or invalid scenario

        ttc_values.append(ttc)

        # Follower-to-follower TTC
        for i in range(1, len(self.followers)):
            leader = self.followers[i - 1]
            follower = self.followers[i]
            relative_velocity = follower.v_x - leader.v_x
            relative_position = leader.x - follower.x - self.desired_gap

            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid scenario

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs

    def fdi_attack(self, follower, step):
        """
        Simulate a more logical FDI attack by gradually altering the follower's state.
        """
        # Gradually increase the follower's velocity
        if 30 < step < 50 and follower.id == 3 :
            attack_intensity = (step - 30) / 20  # Scale from 0 to 1
            follower.v_x += attack_intensity * random.uniform(2, 5)  # Gradual increase in velocity
            print(f"FDI attack on vehicle {follower.id} at step {step}: new velocity {follower.v_x:.2f}")
            
        elif 30 < step < 50 and follower.id == 9 :
            attack_intensity = (step - 30) / 20  # Scale from 0 to 1
            follower.v_x -= attack_intensity * random.uniform(2, 5)  # Gradual increase in velocity
            print(f"FDI attack on vehicle {follower.id} at step {step}: new velocity {follower.v_x:.2f}")

    def run_simulation(self,attack_id):
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
            self.leader.update(a_l, 0)

            # Save leader's position and velocity
            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')

            # Update each follower
            for i, follower in enumerate(self.followers):
                k_p_follower = 1 
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = k_p_follower * distance_to_leader

                # Apply FDI attack
                if follower.id in attack_id:
                    self.fdi_attack(follower, t)


                follower.update(a_f, 0)


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
            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

        # Plotting results
        self.plot_results(x_history, y_history, v_history, min_distances)
        self.plot_ttc_over_time()

    def plot_results(self, x_history, y_history, v_history, min_distances):
        # Plot 1: Trajectory snapshots
        plt.figure(figsize=(15, 8))
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

        # Create a figure with two subplots
        plt.figure(figsize=(10, 12))

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_followers + 1))  # Use viridis colormap
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(np.arange(0, self.time_steps) * self.dt, v_history[i], label=label, color=colors[i], linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(-40, 40)  # Adjusted range to see big spoof effect

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(np.arange(0, self.time_steps) * self.dt, min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_ttc_over_time(self):
        """
        Plot the Time to Collision (TTC) over time.
        """
        time_points = np.arange(len(self.ttc_history)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s )')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()


# Run the simulation
num_followers = 10
attack_id = [3,9]
simulation = LeaderFollowerSimulation(num_followers)
simulation.run_simulation(attack_id)