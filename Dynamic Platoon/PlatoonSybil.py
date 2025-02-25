import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ------------------------------------------------------------------------
# 1) DYNAMIC BICYCLE MODEL (same as before, but can be used by all vehicles)
# ------------------------------------------------------------------------
class DynamicBicycleModel:
    def __init__(self, 
                 x=0.0, 
                 y=0.0, 
                 psi=0.0, 
                 vx=10.0, 
                 vy=0.0, 
                 r=0.0, 
                 dt=0.01,
                 vehicle_id=0):
        self.id  = vehicle_id  # For logging or debug
        # States
        self.x   = x
        self.y   = y
        self.psi = psi
        self.vx  = vx
        self.vy  = vy
        self.r   = r
        
        # Vehicle parameters
        self.m   = 1500.0    # mass [kg]
        self.L_f = 1.2       # CG to front axle [m]
        self.L_r = 1.6       # CG to rear axle [m]
        self.I_z = 2250.0    # Yaw moment of inertia [kg*m^2]
        self.C_f = 19000.0   # Front cornering stiffness [N/rad]
        self.C_r = 20000.0   # Rear cornering stiffness [N/rad]

        self.dt  = dt  # integration step

    def _state_derivs(self, s, a, delta):
        """
        s     = [x, y, psi, vx, vy, r]
        a     = acceleration [m/s^2]
        delta = steering angle [rad]
        """
        x, y, psi, vx, vy, r = s
        vx_safe = max(vx, 0.1)  # clamp to avoid division by zero

        # Slip angles
        alpha_f = delta - np.arctan2((vy + self.L_f*r), vx_safe)
        alpha_r = - np.arctan2((vy - self.L_r*r), vx_safe)

        # Lateral forces (linear tire model)
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # Global kinematics
        x_dot   = vx*np.cos(psi) - vy*np.sin(psi)
        y_dot   = vx*np.sin(psi) + vy*np.cos(psi)
        psi_dot = r

        # Body-frame acceleration
        vx_dot  = a - (F_yf*np.sin(delta))/self.m + vy*r
        vy_dot  = (F_yf*np.cos(delta) + F_yr)/self.m - vx*r
        r_dot   = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z

        return np.array([x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot])

    def update(self, a, delta):
        """
        4th-order Runge-Kutta integration with inputs (a, delta).
        """
        s = np.array([self.x, self.y, self.psi, self.vx, self.vy, self.r])

        k1 = self._state_derivs(s,               a, delta)
        k2 = self._state_derivs(s + 0.5*self.dt*k1, a, delta)
        k3 = self._state_derivs(s + 0.5*self.dt*k2, a, delta)
        k4 = self._state_derivs(s +     self.dt*k3, a, delta)

        s_next = s + (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Unpack
        self.x   = s_next[0]
        self.y   = s_next[1]
        self.psi = s_next[2]
        self.vx  = s_next[3]
        self.vy  = s_next[4]
        self.r   = s_next[5]

    def get_state(self):
        return (self.x, self.y, self.psi, self.vx, self.vy, self.r)


# ------------------------------------------------------------------------
# 2) INNER-LOOP CONTROLLER (per vehicle)
#    - Tracks vx_ref and r_ref, outputs (a, delta)
# ------------------------------------------------------------------------
class InnerLoopController:
    def __init__(self, dt=0.01):
        self.dt = dt

        # Longitudinal Gains (PI)
        self.kp_speed = 1.2
        self.ki_speed = 0.1
        self.speed_error_int = 0.0

        # Lateral Gains: "2-state" feedback on (v_y, r)
        # delta = -K_vy*v_y - K_r*(r - r_ref)  (plus feedforward, if needed)
        self.K_vy   = 0.15
        self.K_r    = 0.08
        self.K_r_ff = 1.0  # feedforward for r_ref

        # Physical saturations
        self.max_steer = 0.5     # rad
        self.min_steer = -0.5
        self.max_accel = 3.0     # m/s^2
        self.min_accel = -5.0

    def control(self, vx, vy, r, vx_ref, r_ref):
        """
        vx, vy, r   : actual states
        vx_ref, r_ref : references
        returns a, delta
        """
        # --- 1) Longitudinal speed control (PI) ---
        speed_err = vx_ref - vx
        self.speed_error_int += speed_err * self.dt

        # Acceleration command
        a_unclamped = (self.kp_speed * speed_err
                       + self.ki_speed * self.speed_error_int)
        a = np.clip(a_unclamped, self.min_accel, self.max_accel)

        # --- 2) Lateral / yaw-rate control ---
        # Simple feedback: delta = -K_vy*v_y - K_r*(r - r_ref)
        delta_unclamped = (
            - self.K_vy * vy
            - self.K_r  * (r - r_ref)
            + self.K_r_ff * r_ref  # optional feedforward
        )
        delta = np.clip(delta_unclamped, self.min_steer, self.max_steer)

        return a, delta


# ------------------------------------------------------------------------
# 3) LEADER-FOLLOWER SIMULATION WITH OUTER + INNER LOOP
# ------------------------------------------------------------------------
class LeaderFollowerSimulationWithInnerLoop:
    def __init__(self, num_followers=5, dt=0.01):
        self.dt = dt
        self.time_steps = int(100.0 / dt)  # 50s sim, for example
        self.num_followers = num_followers

        # Leader
        self.leader = DynamicBicycleModel(x=100, y=10, vx=10, dt=dt, vehicle_id=0)

        # Followers and Controllers
        self.followers = []
        self.controllers = []
        for i in range(num_followers):
            veh = DynamicBicycleModel(x=100 - 14 * (i + 1),
                                      y=10,
                                      vx=3 * (i + 1),
                                      dt=dt,
                                      vehicle_id=i + 1)
            self.followers.append(veh)
            ctrl = InnerLoopController(dt=dt)
            self.controllers.append(ctrl)

        # Outer loop platoon gains
        self.desired_gap = 10.0
        self.k_s = 0.15
        self.k_v = 0.2
        self.k_ey = 0.6
        self.k_epsi = 0.6

        self.road_width = 20
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
                #   Then the relative velocity is (vx_j - vx_i).
                #   If vx_j > vx_i, j is catching up to i => possible collision.
                #   TTC = (dx - desired_gap) / (vx_j - vx_i)
                
                # We'll compute two possible TTCs (i-ahead or j-ahead)
                # and only consider the scenario that is physically valid
                # (positive distance, approaching speed, etc).

                # Scenario A: i ahead, j behind
                if dx > 0:
                    rel_pos = dx - self.desired_gap
                    rel_vel = vehicle_j.vx - vehicle_i.vx
                    if (rel_pos > 0) and (rel_vel > 0):
                        ttc_a = rel_pos / rel_vel
                        ttc_values.append(ttc_a)

                # Scenario B: j ahead, i behind
                else:
                    # dx < 0 => j is ahead, i is behind
                    rel_pos = -dx - self.desired_gap
                    rel_vel = vehicle_i.vx - vehicle_j.vx
                    if (rel_pos > 0) and (rel_vel > 0):
                        ttc_b = rel_pos / rel_vel
                        ttc_values.append(ttc_b)

        # If ttc_values is empty => no risk scenario => return infinity
        if not ttc_values:
            return float('inf')
        else:
            return min(ttc_values)
        
        
    def run_simulation(self):
        # For logging
        time_points = []
        # Histories for leader + followers
        x_history = [[] for _ in range(self.num_followers + 1)]
        y_history = [[] for _ in range(self.num_followers + 1)]
        v_history = [[] for _ in range(self.num_followers + 1)]

        min_distances = []

        # Histories for malicious vehicles
        m_x_history = [[] for _ in range(self.num_malicious)]
        m_y_history = [[] for _ in range(self.num_malicious)]
        m_v_history = [[] for _ in range(self.num_malicious)]

        for step in range(self.time_steps):
            t = step * self.dt
            time_points.append(t)
            # ----------------------------------------------------------
            # 1) Add malicious vehicles after t == 300 (if desired)
            # ----------------------------------------------------------
            if step == 300 and not self.malicious_added:
                self.malicious_vehicles = [
                    DynamicBicycleModel(
                        x=60 - 10 * (i + 1),
                        y=10,
                        vx= 4*(i+1),
                        vehicle_id=100 + i
                    )
                    for i in range(self.num_malicious)
                ]
                print('done')
                self.malicious_added = True

            # Leader Control
            a_leader = 0.5 * (6.0 - self.leader.vx)  # Simple speed control
            delta_leader = 0.0
            self.leader.update(a_leader, delta_leader)

            x_history[0].append(self.leader.x)
            y_history[0].append(self.leader.y)
            v_history[0].append(self.leader.vx)
            # Follower Vehicles
            min_dist_timestep = float('inf')
            for i, follower in enumerate(self.followers):
                if i == 0:
                    x0, y0, psi0, vx0, vy0, r0 = self.leader.get_state()
                else:
                    x0, y0, psi0, vx0, vy0, r0 = self.followers[i - 1].get_state()

                x1, y1, psi1, vx1, vy1, r1 = follower.get_state()

                dx = x1 - x0
                dy = y1 - y0
                cos_psi0 = np.cos(psi0)
                sin_psi0 = np.sin(psi0)
                e_x = cos_psi0 * dx + sin_psi0 * dy
                e_y = -sin_psi0 * dx + cos_psi0 * dy

                e_s = -e_x - self.desired_gap
                e_psi = psi0 - psi1

                vx_ref = vx0 + self.k_s * (e_s) + self.k_v * (vx0 - vx1)
                r_ref = r0 + self.k_ey * e_y + self.k_epsi * e_psi

                a_f, delta_f = self.controllers[i].control(vx1, vy1, r1, vx_ref, r_ref)


                follower.update(a_f, delta_f)


                x_history[i + 1].append(follower.x)
                y_history[i + 1].append(follower.y)
                v_history[i + 1].append(follower.vx)
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
                    m_v_history[i_m].append(malicious.vx)

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
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)


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
        plt.ylim(-40, 60)

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
        plt.ylim(-1,10)
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------------
# 4) MAIN SCRIPT TO RUN
# ------------------------------------------------------------------------
if __name__ == "__main__":
    sim = LeaderFollowerSimulationWithInnerLoop(num_followers=5, dt=0.01)
    sim.run_simulation()
