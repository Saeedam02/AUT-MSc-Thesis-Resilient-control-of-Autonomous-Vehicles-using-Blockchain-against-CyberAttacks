from time import time
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class DynamicBicycleModel:
    def __init__(self, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.psi = psi  # Yaw angle
        self.v_x = vx  # Longitudinal velocity
        self.v_y = vy  # Lateral velocity
        self.r = r    # Yaw rate

        # Vehicle-specific parameters
        self.m = 1500     # Vehicle mass (kg)
        self.L_f = 1.2    # Distance from CG to front axle (m)
        self.L_r = 1.6    # Distance from CG to rear axle (m)
        self.I_z = 2250   # Yaw moment of inertia (kg*m^2)
        self.C_f = 19000  # Cornering stiffness front (N/rad)
        self.C_r = 20000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01    # Time step (s)

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

        self.x  = xx[0]
        self.y  = xx[1]
        self.psi = xx[2]
        self.v_x = xx[3]
        self.v_y = xx[4]
        self.r   = xx[5]

    def f(self, xx, a, delta):
        """
        calculating the states' derivatives using the vehicle's dynamics
        """
        x    = xx[0]
        y    = xx[1]
        psi  = xx[2]
        v_x  = xx[3]
        v_y  = xx[4]
        r    = xx[5]

        # Calculate slip angles
        alpha_f = delta - np.arctan2((v_y + self.L_f * r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r * r), v_x)

        # Calculate lateral forces
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # State derivatives
        x_dot     = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot     = v_x * np.sin(psi) + v_y * np.cos(psi)
        psi_dot   = r
        v_x_dot   = a - (F_yf * np.sin(delta)) / self.m + v_y * r
        v_y_dot   = (F_yf * np.cos(delta) + F_yr) / self.m - v_x * r
        r_dot     = (self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr) / self.I_z

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
        # Leader vehicle
        self.leader = DynamicBicycleModel(id=0, x=100, y=10, vx=6.0)

        # Followers
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
        self.road_width = 20  # Width of the road (m)
        self.ttc_history = []  # Track minimum TTC over time

        # ---- Malicious (Sybil) attack additions ----
        self.malicious_vehicles = []
        self.num_malicious = 3      # number of malicious vehicles
        self.malicious_added = False
        # --------------------------------------------

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) among *all* vehicles in the network,
        including malicious vehicles if they exist.

        We'll return the minimum TTC found across all vehicle pairs.
        """
        # 1) Build a master list of all vehicles
        all_vehicles = [self.leader] + self.followers
        if self.malicious_added:
            all_vehicles += self.malicious_vehicles

        ttc_values = []

        # 2) Pairwise comparison
        for i in range(len(all_vehicles)):
            for j in range(i + 1, len(all_vehicles)):
                vehicle_i = all_vehicles[i]
                vehicle_j = all_vehicles[j]

                # -- Example: consider purely longitudinal collisions --
                # If you only care about collisions along x-axis:

                # We'll define "i" as the car in front, "j" as the one behind.
                # We check both directions because we don't know which is ahead:
                dx = vehicle_i.x - vehicle_j.x
                # For convenience, define the relative gap as:
                #   dx - desired_gap
                # Then define velocity difference.
                
                # Let’s check if vehicle_i is ahead (dx > 0):
                #   That means j is behind i. 
                #   Then the relative velocity is (v_x_j - v_x_i).
                #   If v_x_j > v_x_i, j is catching up to i => possible collision.
                #   TTC = (dx - desired_gap) / (v_x_j - v_x_i)
                
                # We'll compute two possible TTCs (i-ahead or j-ahead)
                # and only consider the scenario that is physically valid
                # (positive distance, approaching speed, etc).

                # Scenario A: i ahead, j behind
                if dx > 0:
                    rel_pos = dx - self.desired_gap
                    rel_vel = vehicle_j.v_x - vehicle_i.v_x
                    if (rel_pos > 0) and (rel_vel > 0):
                        ttc_a = rel_pos / rel_vel
                        ttc_values.append(ttc_a)

                # Scenario B: j ahead, i behind
                else:
                    # dx < 0 => j is ahead, i is behind
                    rel_pos = -dx - self.desired_gap
                    rel_vel = vehicle_i.v_x - vehicle_j.v_x
                    if (rel_pos > 0) and (rel_vel > 0):
                        ttc_b = rel_pos / rel_vel
                        ttc_values.append(ttc_b)

        # If ttc_values is empty => no risk scenario => return infinity
        if not ttc_values:
            return float('inf')
        else:
            return min(ttc_values)

    def run_simulation(self):
        # Histories for leader + followers
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]

        # Histories for malicious vehicles
        m_x_history = [[] for _ in range(self.num_malicious)]
        m_y_history = [[] for _ in range(self.num_malicious)]
        m_v_history = [[] for _ in range(self.num_malicious)]

        min_distances = []
        time_points = np.arange(0, self.time_steps) * self.dt

        for t in range(self.time_steps):
            # ----------------------------------------------------------
            # 1) Add malicious vehicles after t == 300 (if desired)
            # ----------------------------------------------------------
            if t == 300 and not self.malicious_added:
                self.malicious_vehicles = [
                    DynamicBicycleModel(
                        id=100 + i,
                        x=60 - 10 * (i + 1),
                        y=10,
                        vx= 8*(i+1)
                    )
                    for i in range(self.num_malicious)
                ]
                self.malicious_added = True

            # ----------------------------------------------------------
            # 2) Update leader
            # ----------------------------------------------------------
            v_target = 6.0
            k_p = 1.0
            a_l = k_p * (v_target - self.leader.v_x)
            self.leader.update(0, 0)

            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.v_x)

            # ----------------------------------------------------------
            # 3) Update followers
            # ----------------------------------------------------------
            for i, follower in enumerate(self.followers):
                k_p_follower = 1
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = k_p_follower * distance_to_leader
                follower.update(0, 0)

                x_history[i + 1].append(follower.x)
                y_history[i + 1].append(follower.y)
                v_history[i + 1].append(follower.v_x)

            # ----------------------------------------------------------
            # 4) Update malicious vehicles (if added)
            # ----------------------------------------------------------
            if self.malicious_added:
                for i_m, malicious in enumerate(self.malicious_vehicles):
                    k_p_malicious = 1
                    distance_to_leader = self.leader.x - malicious.x - self.desired_gap * (i + 1)
                    a_m = k_p_malicious * distance_to_leader
                    delta_m = 0.0
                    malicious.update(a_m, delta_m)

                    m_x_history[i_m].append(malicious.x)
                    m_y_history[i_m].append(malicious.y)
                    m_v_history[i_m].append(malicious.v_x)

            # ----------------------------------------------------------
            # 5) Calculate minimum distance among *all* vehicles
            # ----------------------------------------------------------
            # Gather all vehicles: leader, followers, and malicious
            all_vehicles = [self.leader] + self.followers
            if self.malicious_added:
                all_vehicles.extend(self.malicious_vehicles)

            min_dist_timestep = float('inf')
            for i_v in range(len(all_vehicles)):
                for j_v in range(i_v + 1, len(all_vehicles)):
                    dx = all_vehicles[i_v].x - all_vehicles[j_v].x
                    dy = all_vehicles[i_v].y - all_vehicles[j_v].y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < min_dist_timestep:
                        min_dist_timestep = dist

            min_distances.append(min_dist_timestep)
            # ----------------------------------------------------------------
            # 5) Calculate and store TTC
            # ----------------------------------------------------------------
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

        #print("TTC history:", self.ttc_history)

        # --------------------------------------------------------------------
        # Plot #1: Vehicle positions at specific snapshots in time
        # --------------------------------------------------------------------
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps * 0.01), 
            int(self.time_steps * 0.2), 
            int(self.time_steps * 0.4),
            int(self.time_steps * 0.6), 
            int(self.time_steps * 0.8), 
            int(self.time_steps * 0.99)
        ]
        
        for idx, sample_idx in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            # Leader + followers
            for i in range(self.num_followers + 1):
                plt.plot(x_history[i][sample_idx], 
                         y_history[i][sample_idx], 
                         '.k' if i > 0 else '.r', 
                         markersize=10 if i == 0 else 5)

            # Malicious vehicles (if they exist at that time)
            if self.malicious_added and sample_idx >= 300:
                # malicious data starts at t=300 -> index offset = sample_idx - 300
                offset = sample_idx - 300
                for i_m in range(self.num_malicious):
                    if offset < len(m_x_history[i_m]):
                        plt.plot(m_x_history[i_m][offset], 
                                 m_y_history[i_m][offset], 
                                 'Xr', markersize=8, label="Malicious" if i_m == 0 else "")

            # Road boundaries
            plt.plot([0, self.leader.x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, self.leader.x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={sample_idx * self.dt:.2f} sec')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

        # --------------------------------------------------------------------
        # Plot #2: Velocity consensus (leader + followers) and Min Distance
        # --------------------------------------------------------------------
        plt.figure(figsize=(10, 12))

        # Subplot for velocities
        plt.subplot(2, 1, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_followers + 1))
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(time_points, v_history[i], label=label, color=colors[i], linewidth=2.5)

        # Also plot malicious vehicles velocities
        if self.malicious_added:
            mal_colors = plt.cm.autumn(np.linspace(0, 1, self.num_malicious))
            # Malicious vehicles only have data from step 300 => time from 300*dt to end
            time_points_mal = time_points[300:]  # same dt, just offset in index
            for i_m in range(self.num_malicious):
                if len(m_v_history[i_m]) > 0:
                    plt.plot(time_points_mal, m_v_history[i_m],
                             label=f'Malicious {i_m+1}',
                             color=mal_colors[i_m],
                             linewidth=2.5,
                             linestyle='--')

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(-20, 40)

        # Subplot for minimum distance
        plt.subplot(2, 1, 2)
        plt.plot(time_points, min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)
        plt.ylim(-1, 25)
        plt.tight_layout()
        plt.show()

        # --------------------------------------------------------------------
        # Plot #3: Time to Collision (TTC)
        # --------------------------------------------------------------------
        self.plot_ttc_over_time()

    def plot_ttc_over_time(self):
        time_points = np.arange(len(self.ttc_history)) * self.dt
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC = 1s')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC = 2s')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.title('Time to Collision (TTC) Over Time')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()


# Run the simulation
num_followers = 10
simulation = LeaderFollowerSimulation(num_followers)
simulation.run_simulation()
