import numpy as np
import matplotlib.pyplot as plt

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
#    - Tracks v_x_ref and r_ref, outputs (a, delta)
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
        # Leader (we can choose to use or skip an inner loop here)
        self.leader = DynamicBicycleModel(x=100, y=10, vx= 10, dt=dt, vehicle_id=0)
        
        # For simplicity, let's do direct control for leader:
        self.leader_speed_ref = 6.0  # We'll keep it constant
        self.kp_leader = 1.0
        self.road_width = 20
        self.ttc_history = []  # Track minimum TTC over time
        # Follower vehicles, each with its own inner controller
        self.followers = []
        self.controllers = []
        for i in range(num_followers):
            veh = DynamicBicycleModel(x=100 - 14*(i+1),
                                      y=10,
                                      vx=3*(i+1),
                                      dt=dt,
                                      vehicle_id=i+1)
            self.followers.append(veh)
            ctrl = InnerLoopController(dt=dt)
            self.controllers.append(ctrl)

        # Outer loop platoon gains
        self.desired_gap = 10.0
        self.k_s   = 0.15   # how strongly we correct spacing error -> v_x_ref
        self.k_v   = 0.2   # optionally might do relative speed
        self.k_ey  = 0.6   # lateral offset -> r_ref
        self.k_epsi= 0.6   # heading error -> r_ref
    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        
        # Leader-to-first-follower TTC
        leader = self.leader
        first_follower = self.followers[0]
        relative_velocity = first_follower.vx - leader.vx
        relative_position = leader.x - first_follower.x 

        if relative_position > 0 and relative_velocity > 0:
            ttc = relative_position / relative_velocity
        else:
            ttc = float('inf')  # No collision risk or invalid scenario

        ttc_values.append(ttc)

        # Follower-to-follower TTC
        for i in range(1, len(self.followers)):
            leader = self.followers[i - 1]
            follower = self.followers[i]
            relative_velocity = follower.vx - leader.vx
            relative_position = leader.x - follower.x 
            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid scenario

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs


    def run_simulation(self):
        # For logging
        time_points = []
        leader_xhist, leader_yhist, leader_vxhist = [], [], []
        follower_xhist = [[] for _ in self.followers]
        follower_yhist = [[] for _ in self.followers]
        follower_vxhist= [[] for _ in self.followers]
        min_distances = []

        for step in range(self.time_steps):
            t = step * self.dt
            time_points.append(t)

            #-----------------------------------------------
            # Leader Control (just do a simple speed P-control)
            #-----------------------------------------------

            a_leader = self.kp_leader*(self.leader_speed_ref - self.leader.vx)
            # keep steering = 0 for leader (straight)
            delta_leader = 0.0

            self.leader.update(a_leader, delta_leader)
            #if t == 0: print("Initial leader state:", self.leader.get_state())

            leader_xhist.append(self.leader.x)
            leader_yhist.append(self.leader.y)
            leader_vxhist.append(self.leader.vx)

            #-----------------------------------------------
            # Follower Vehicles
            #-----------------------------------------------
            min_dist_timestep = float('inf')

            for i, follower in enumerate(self.followers):
                # Outer loop: measure spacing/lateral error w.r.t. the preceding vehicle
                if i == 0:
                    # preceding = leader
                    x0, y0, psi0, vx0, vy0, r0 = self.leader.get_state()
                else:
                    x0, y0, psi0, vx0, vy0, r0 = self.followers[i-1].get_state()

                x1, y1, psi1, vx1, vy1, r1 = follower.get_state()

                # transform follower's position into preceding vehicle's frame
                dx = x1 - x0
                dy = y1 - y0

                cos_psi0 = np.cos(psi0)
                sin_psi0 = np.sin(psi0)
                e_x =  cos_psi0*dx + sin_psi0*dy
                e_y = -sin_psi0*dx + cos_psi0*dy

                # spacing error: we want the follower to be 'desired_gap' behind
                # so let's define e_s = (x0 - x1) - desired_gap  in the global sense
                # but we've done a frame rotation, so we can also do:
                e_s = -e_x - self.desired_gap  # i.e. if e_x < -10, we are behind

                # heading error for outer loop
                e_psi = psi0 - psi1

                # For a simple approach, we'll generate references:
                # v_x_ref = vx0 + k_s*(e_s)    (follower tries to match leader's speed plus correct spacing)
                v_x_ref = vx0 + self.k_s * (e_s)+ self.k_v * (vx0 - vx1)

                # r_ref = r0 + k_ey*e_y + k_epsi*e_psi
                # or simpler: just use the lateral error
                r_ref  = r0 + self.k_ey*e_y + self.k_epsi*e_psi

                #--- Now call the follower's inner loop controller
                a_f, delta_f = self.controllers[i].control(
                    vx1, vy1, r1,
                    v_x_ref, r_ref
                )

                #--- Update follower dynamics
                follower.update(a_f, delta_f)

                #--- For logging
                follower_xhist[i].append(follower.x)
                follower_yhist[i].append(follower.y)
                follower_vxhist[i].append(follower.vx)

                # Distance to preceding vehicle (Euclidean)
                dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                if dist < min_dist_timestep:
                    min_dist_timestep = dist

            # Keep track of min distance among all follower pairs for plotting
            min_distances.append(min_dist_timestep)
            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)
        #-------------------------------------------------------
        # After simulation, plot results
        #-------------------------------------------------------
        # self.plot_results(time_points, 
        #                   leader_xhist, leader_yhist, leader_vxhist, 
        #                   follower_xhist, follower_yhist, follower_vxhist)
        all_vxhist = [leader_vxhist] + follower_vxhist  
        self.plot_trajectory([leader_xhist] + follower_xhist, [leader_yhist] + follower_yhist)
        self.plot_velocity_and_distance(time_points, all_vxhist, min_distances)
        self.plot_ttc_over_time(time_points)

    def plot_ttc_over_time(self,time_points):
        """
        Plot the Time to Collision (TTC) over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s)')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.ylim(-1,10)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self, x_history, y_history):
        """
        Plots snapshots of the vehicles along the road.
        """
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps * 0.01), 
            int(self.time_steps * 0.2), 
            int(self.time_steps * 0.4),
            int(self.time_steps * 0.6),
            int(self.time_steps * 0.8),
            int(self.time_steps * 0.99)
        ]

        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers + 1):  # This is correct
                if i == 0:  # leader
                    if t < len(x_history[i]):  # Check if t is within bounds
                        plt.plot(x_history[i][t], y_history[i][t], '.r', markersize=10)
                else:
                    if t < len(x_history[i]):  # Check if t is within bounds
                        plt.plot(x_history[i][t], y_history[i][t], '.k', markersize=5)
            # Draw some road boundaries
            leader_xmax = max(x_history[0])
            plt.plot([0, leader_xmax+50],[0,0], 'b-', linewidth=2)
            plt.plot([0, leader_xmax+50],[self.road_width,self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t*self.dt:.2f} s')
            plt.ylim(-5, self.road_width+5)
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_results(self, time_points, 
                     leader_xhist, leader_yhist, leader_vxhist,
                     follower_xhist, follower_yhist, follower_vxhist):
        plt.figure(figsize=(12,6))
        # (a) X-Y Trajectories
        plt.subplot(1,2,1)
        plt.plot(leader_xhist, leader_yhist, 'r-', label='Leader')
        for i in range(len(self.followers)):
            plt.plot(follower_xhist[i], follower_yhist[i], '-o', ms=2, label=f'Follower {i+1}')
        plt.legend()
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('X-Y Trajectory')
        plt.grid(True)
        plt.axis('equal')

        # (b) Speed Over Time
        plt.subplot(1,2,2)
        plt.plot(time_points, leader_vxhist, 'r-', label='Leader vx')
        for i in range(len(self.followers)):
            plt.plot(time_points, follower_vxhist[i], label=f'Follower {i+1} vx')
        plt.xlabel('Time [s]')
        plt.ylabel('Longitudinal speed [m/s]')
        plt.title('Speed vs. Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


    def plot_velocity_and_distance(self, time_points, v_history, min_distances):
        """
        Plots (a) Velocity of each vehicle, (b) Minimum distance among vehicles over time.
        """
        plt.figure(figsize=(10, 12))

        #--- (1) Velocity Subplot
        plt.subplot(2, 1, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_followers + 1))
        for i in range(self.num_followers + 1):  # This is correct
            if i < len(v_history):  # Check if i is within bounds
                if i == 0:
                    plt.plot(time_points, v_history[i], color=colors[i], label='Leader', linewidth=2)
                else:
                    plt.plot(time_points, v_history[i], color=colors[i], label=f'Follower {i}', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)
        plt.legend(loc='best')

        #--- (2) Distance Subplot
        plt.subplot(2, 1, 2)
        plt.plot(time_points, min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.grid(True)
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------------
# 4) MAIN SCRIPT TO RUN
# ------------------------------------------------------------------------
if __name__ == "__main__":
    sim = LeaderFollowerSimulationWithInnerLoop(num_followers=5, dt=0.01)
    sim.run_simulation()
