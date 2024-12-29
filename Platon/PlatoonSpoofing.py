import numpy as np
import matplotlib.pyplot as plt
import random

###############################################################################
# DynamicBicycleModel Class (Naive, No Blockchain)
###############################################################################
class DynamicBicycleModel:
    """
    Simple bicycle model for each vehicle with no security checks.
    """
    def __init__(self, vehicle_id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.psi = psi
        self.v_x = vx
        self.v_y = vy
        self.r = r

        # Physical parameters
        self.m = 1500
        self.L_f = 1.2
        self.L_r = 1.6
        self.I_z = 2250
        self.C_f = 19000
        self.C_r = 20000
        self.dt = 0.01

        # For plotting
        self.velocity_history = []

    def update(self, a, delta):
        """
        Standard 4th-order Runge-Kutta bicycle model integration.
        """
        xx = np.array([self.x, self.y, self.psi, self.v_x, self.v_y, self.r])
        k1 = self.f(xx, a, delta)
        k2 = self.f(xx + self.dt/2*k1, a, delta)
        k3 = self.f(xx + self.dt/2*k2, a, delta)
        k4 = self.f(xx + self.dt*k3, a, delta)
        xx = xx + self.dt*(k1 + 2*k2 + 2*k3 + k4)/6

        self.x, self.y, self.psi, self.v_x, self.v_y, self.r = xx
        self.velocity_history.append(self.v_x)

    def f(self, xx, a, delta):
        x, y, psi, v_x, v_y, r = xx
        alpha_f = delta - np.arctan2((v_y + self.L_f*r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r*r), v_x)
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        x_dot = v_x*np.cos(psi) - v_y*np.sin(psi)
        y_dot = v_x*np.sin(psi) + v_y*np.cos(psi)
        psi_dot = r
        v_x_dot = a - (F_yf*np.sin(delta))/self.m + v_y*r
        v_y_dot = (F_yf*np.cos(delta) + F_yr)/self.m - v_x*r
        r_dot = (self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr)/self.I_z
        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def get_state(self):
        return {
            'id': self.id,
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }

###############################################################################
# Spoofing Attack (Naive)
###############################################################################
def simulate_spoofing_attack(fleet, attacker_id, target_id):
    """
    The attacker modifies the target vehicle's velocity or states 
    by directly overwriting them in the naive shared data model.
    This attack succeeds because there's no cryptographic or blockchain validation.
    """
    attacker = None
    target = None
    for v in fleet:
        if v.id == attacker_id:
            attacker = v
        elif v.id == target_id:
            target = v

    if attacker and target:
        print(f"\n[!] Spoofing Attack: Vehicle {attacker_id} impersonates Vehicle {target_id}.")
        # Overwrite the target's velocity with a large/unrealistic value
        fake_vx = attacker.v_x + 20  # some big jump
        print(f"    Fake velocity set to {fake_vx:.2f} m/s for Vehicle {target_id}.")
        target.v_x = fake_vx
    else:
        print("[!] Spoofing Attack Failed: Attacker or Target not found.")

###############################################################################
# Leader-Follower Simulation (Naive)
###############################################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers):
        """
        Leader at (x=100, y=10), each follower behind it. 
        No blockchain: states are only local to each vehicle object.
        """
        # Create leader
        self.leader = DynamicBicycleModel(vehicle_id=0, x=100, y=10, vx=6.0)

        # Create followers
        self.followers = [
            DynamicBicycleModel(vehicle_id=i+1, x=100 - 10*(i+1), y=10, vx=6.0)
            for i in range(num_followers)
        ]
        self.vehicles = [self.leader] + self.followers

        self.desired_gap = 10
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        # For plotting
        self.x_history = [[] for _ in range(num_followers+1)]
        self.y_history = [[] for _ in range(num_followers+1)]
        self.v_history = [[] for _ in range(num_followers+1)]
        self.min_distances = []
        self.ttc_history = []  # Track minimum TTC over time
        self.time_points = np.arange(0, self.time_steps)*self.dt

    def calculate_ttc(self):
        """
        Calculate Time to Collision (TTC) for all vehicles.
        """
        ttc_values = []
        for i in range(1, len(self.vehicles)):  # Start from the first follower
            leader = self.vehicles[i - 1]
            follower = self.vehicles[i]

            relative_velocity = follower.v_x - leader.v_x
            relative_position = leader.x - follower.x - self.desired_gap

            # Calculate TTC only if conditions are valid
            if relative_position > 0 and relative_velocity > 0:
                ttc = relative_position / relative_velocity
            else:
                ttc = float('inf')  # No collision risk or invalid position

            ttc_values.append(ttc)

        return min(ttc_values)  # Return the minimum TTC across all pairs


    def run_simulation(self, spoof_params=None):
        """
        A naive run: if we pass 'spoof_params', we do a spoofing attack at some step.
        """
        spoof_step = -1
        attacker_id = None
        target_id = None
        if spoof_params:
            spoof_step = spoof_params.get('step', -1)
            attacker_id = spoof_params.get('attacker_id', None)
            target_id = spoof_params.get('target_id', None)

        for step in range(self.time_steps):
            print('Step:', step)
            # 1) Leader update
            v_target = 6.0
            kp = 1.0
            a_l = kp*(v_target - self.leader.v_x)
            self.leader.update(a_l, 0)

            # Save leader state
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)
            self.v_history[0].append(self.leader.v_x)

            min_dist = float('inf')

            # 2) Followers update
            for i, follower in enumerate(self.followers):
                distance_to_leader = self.leader.x - follower.x - self.desired_gap*(i+1)
                a_f = 1.0 * distance_to_leader
                follower.update(a_f, 0)

                self.x_history[i+1].append(follower.x)
                self.y_history[i+1].append(follower.y)
                self.v_history[i+1].append(follower.v_x)

                # Distance calculation
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x)**2 + (follower.y - self.leader.y)**2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i-1].x)**2 + 
                                   (follower.y - self.followers[i-1].y)**2)
                min_dist = min(min_dist, dist)

            self.min_distances.append(min_dist)

            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)

            # 3) Possibly do a spoofing attack at a specific step
            if spoof_step < step < spoof_step + 5 and attacker_id is not None and target_id is not None:
                simulate_spoofing_attack(self.vehicles, attacker_id, target_id)
        #print(self.ttc_history)
        # After the simulation, plot results
        self.plot_trajectory_snapshots()
        self.plot_velocity_and_min_distance()  # Updated function call
        self.plot_ttc_over_time()

    def plot_ttc_over_time(self):
        """
        Plot the Time to Collision (TTC) over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s)')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.title('Time to Collision (TTC) Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trajectory_snapshots(self):
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps*0.01), int(self.time_steps*0.2),
            int(self.time_steps*0.4),  int(self.time_steps*0.6),
            int(self.time_steps*0.8),  int(self.time_steps*0.99)
        ]
        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(len(self.vehicles)):
                if i == 0:
                    plt.plot(self.x_history[i][t], self.y_history[i][t],
                             '.r', markersize=10)  # leader in red
                else:
                    plt.plot(self.x_history[i][t], self.y_history[i][t],
                             '.k', markersize=5)
            leader_x = max(self.x_history[0]) if self.x_history[0] else 100
            plt.plot([0, leader_x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t*self.dt:.2f}s')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

    def plot_velocity_and_min_distance(self):
        """
        Plot Velocity Consensus and Minimum Distance in one figure.
        """
        plt.figure(figsize=(10, 12))

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.vehicles)))  # Use viridis colormap
        for i, veh in enumerate(self.vehicles):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(self.time_points, self.v_history[i], label=label, color=colors[i], linewidth=2.0)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(-40, 40)  # Adjusted range to see big spoof effect

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(self.time_points, self.min_distances, 'b-', label='Min Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
###############################################################################
# Main Execution
###############################################################################
if __name__ == "__main__":
    # Create a naive simulation with 3 followers
    num_followers = 10
    simulation = LeaderFollowerSimulation(num_followers)

    # Define a single spoofing event at step=30 
    # Attacker = vehicle_id=2 tries to impersonate (modify state of) target=vehicle_id=1
    spoof_params = {
        'step': 30,
        'attacker_id': 2,
        'target_id': 1
    }

    # Run the naive simulation with a spoofing attack
    simulation.run_simulation(spoof_params)
