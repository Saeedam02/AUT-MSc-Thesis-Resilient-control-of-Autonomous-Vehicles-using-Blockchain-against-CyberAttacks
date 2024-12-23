import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque
import logging
from datetime import datetime

##################################################
# Logging Configuration
##################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'platooning_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)

##################################################
# Security Monitor from First Script
##################################################
class SecurityMonitor:
    def __init__(self):
        self.message_rates = {}
        self.velocity_changes = {}
        self.window_size = 50  # Time window for rate monitoring

    def track_message_rate(self, vehicle_id, current_queue_size):
        if vehicle_id not in self.message_rates:
            self.message_rates[vehicle_id] = deque(maxlen=self.window_size)
        self.message_rates[vehicle_id].append(current_queue_size)

    def track_velocity_change(self, vehicle_id, velocity):
        if vehicle_id not in self.velocity_changes:
            self.velocity_changes[vehicle_id] = deque(maxlen=self.window_size)
        self.velocity_changes[vehicle_id].append(velocity)

    def detect_anomalies(self, vehicle_id):
        anomalies = []

        # Check message rate anomalies
        if len(self.message_rates.get(vehicle_id, [])) >= 2:
            rate = np.mean(list(self.message_rates[vehicle_id]))
            # Example threshold for suspicious message rates
            if rate > 100:
                anomalies.append(f"High message rate: {rate:.2f}")

        # Check velocity anomalies
        if len(self.velocity_changes.get(vehicle_id, [])) >= 2:
            velocities = list(self.velocity_changes[vehicle_id])
            velocity_change = abs(velocities[-1] - velocities[-2])
            # Example threshold for suspicious velocity changes
            if velocity_change > 20:
                anomalies.append(f"Suspicious velocity change: {velocity_change:.2f}")

        return anomalies


##################################################
# Dynamic Bicycle Model + Security Additions
##################################################
class DynamicBicycleModel:
    """
    Extended bicycle model with a message queue and security features
    to demonstrate DDoS/flood attacks and anomaly detection.
    """
    def __init__(self, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0):
        self.id = id  # Vehicle ID
        # States
        self.x = x  # X position
        self.y = y  # Y position
        self.psi = psi  # Yaw angle
        self.v_x = vx  # Longitudinal velocity
        self.v_y = vy  # Lateral velocity
        self.r = r   # Yaw rate

        # Message queue for simulating network traffic
        self.message_queue = deque(maxlen=1000)  # limit queue size
        self.processing_delay = 0.001
        self.last_update_time = time.time()

        # Flags for anomaly and recovery
        self.recovery_mode = False

        # Vehicle-specific parameters (same as original)
        self.m = 1500      # Vehicle mass kg
        self.L_f = 1.2     # Distance from CG to front axle (m)
        self.L_r = 1.6     # Distance from CG to rear axle (m)
        self.I_z = 2250    # Yaw moment of inertia (kg*m^2)
        self.C_f = 19000   # Cornering stiffness front (N/rad)
        self.C_r = 20000   # Cornering stiffness rear (N/rad)
        self.dt = 0.01     # Time step (s)

        # Used to store velocity history for plotting
        self.velocity_history = []

    def update_dynamics(self, security_monitor):
        """
        High-level update that processes the message queue,
        checks for anomalies, and adjusts control if in recovery mode.
        Then it calls the standard bicycle model update.
        """
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process messages up to a small time budget to simulate CPU limits
        start_time = time.time()
        messages_processed = 0
        while self.message_queue and (time.time() - start_time) < 0.1:
            self.message_queue.popleft()
            messages_processed += 1
            time.sleep(self.processing_delay)

        # Track queue size and velocity in security monitor
        queue_size = len(self.message_queue)
        security_monitor.track_message_rate(self.id, queue_size)
        security_monitor.track_velocity_change(self.id, self.v_x)

        # Detect anomalies
        anomalies = security_monitor.detect_anomalies(self.id)
        if anomalies:
            logging.warning(f"Vehicle {self.id} - Anomalies detected: {anomalies}")
            self.recovery_mode = True

        # If in recovery mode, apply simpler control
        if self.recovery_mode:
            self.apply_recovery_control()
        else:
            # Normal update with no special control input (or put your control logic here)
            pass

        # Record velocity for plotting
        self.velocity_history.append(self.v_x)

    def apply_recovery_control(self):
        """
        Example of a simplistic recovery: reduce velocity if anomalies are detected.
        """
        self.v_x *= 0.95  # Gradually slow down if under attack or anomaly

    ##################################################
    # Original (unmodified) dynamic bicycle code
    ##################################################
    def update(self, a, delta):
        """
        Standard bicycle model update using Runge-Kutta integration.
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
        self.r = xx[5]    # Yaw rate

    def f(self, xx, a, delta):
        """
        State derivatives.
        """
        x = xx[0]
        y = xx[1]
        psi = xx[2]
        v_x = xx[3]
        v_y = xx[4]
        r = xx[5]

        alpha_f = delta - np.arctan2((v_y + self.L_f * r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r * r), v_x)

        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
        psi_dot = r
        v_x_dot = a - (F_yf * np.sin(delta)) / self.m + v_y * r
        v_y_dot = (F_yf * np.cos(delta) + F_yr) / self.m - v_x * r
        r_dot = (self.L_f * F_yf * np.cos(delta) - self.L_r * F_yr) / self.I_z

        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])

    def get_state(self):
        """
        Get the current state of the vehicle for logging or debugging.
        """
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }


##################################################
# Attack Simulation (flood, velocity manipulation, etc.)
##################################################
def simulate_network_attack(vehicles, target_ids, intensity=1000, duration=20, attack_type="flood"):
    """
    Simulate various types of network attacks against the message queue.
    """
    for vehicle in vehicles:
        if vehicle.id in target_ids:
            if attack_type == "flood":
                # Simulate message flood
                flood_messages = [
                    {'timestamp': time.time(), 'type': 'status_update'}
                    for _ in range(intensity)
                ]
                vehicle.message_queue.extend(flood_messages)
                logging.info(f"Attacking Vehicle {vehicle.id} with {intensity} messages (flood). Queue size now {len(vehicle.message_queue)}")
            elif attack_type == "velocity":
                # Simulate velocity manipulation
                delta_v = random.uniform(20, 50)
                vehicle.v_x += delta_v
                logging.info(f"Attacking Vehicle {vehicle.id} with velocity manipulation: +{delta_v:.2f} m/s")


##################################################
# Leader-Follower (Platooning) Simulation
##################################################
class LeaderFollowerSimulation:
    def __init__(self, num_followers):
        # Initialize leader
        self.leader = DynamicBicycleModel(id=0, x=100, y=10, vx=6.0)
        # Initialize followers
        self.followers = [
            DynamicBicycleModel(
                id=i + 1,
                x=100 - 10 * (i + 1),
                y=10,
                vx=6.0
            )
            for i in range(num_followers)
        ]
        self.vehicles = [self.leader] + self.followers  # combined list

        self.num_followers = num_followers
        self.desired_gap = 10   # Desired gap (m)
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        # For plotting
        self.x_history = [[] for _ in range(num_followers + 1)]
        self.y_history = [[] for _ in range(num_followers + 1)]
        self.v_history = [[] for _ in range(num_followers + 1)]
        self.min_distances = []
        self.time_points = np.arange(0, self.time_steps) * self.dt

        # Security monitor for all vehicles
        self.security_monitor = SecurityMonitor()

    def run_simulation(self, attack_params=None):
        """
        Main simulation loop with optional DDoS/flood attack.
        """
        for t_step in range(self.time_steps):
            logging.info(f"\n--- Simulation Step {t_step} ---")

            # If we are in the attack window, simulate the attack
            if attack_params:
                if (t_step >= attack_params['start']) and (t_step <= attack_params['end']):
                    simulate_network_attack(
                        self.vehicles,
                        attack_params['targets'],
                        attack_params.get('intensity', 500),
                        attack_type=attack_params.get('type', 'flood')
                    )

            # 1) Update Leader
            v_target = 6.0
            k_p = 1
            a_l = k_p * (v_target - self.leader.v_x)
            # Security-based update
            self.leader.update_dynamics(self.security_monitor)
            # Physical model update
            self.leader.update(a_l, 0)

            # 2) Save Leader's position and velocity
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)
            self.v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')

            # 3) Update Followers
            for i, follower in enumerate(self.followers):
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = 1.0 * distance_to_leader  # simple proportional control

                # Security-based update
                follower.update_dynamics(self.security_monitor)
                # Physical model update
                follower.update(a_f, 0)

                # Save to history
                self.x_history[i + 1].append(follower.x)
                self.y_history[i + 1].append(follower.y)
                self.v_history[i + 1].append(follower.v_x)

                # Compute distance from this follower to the one in front
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x) ** 2 + (follower.y - self.leader.y) ** 2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i - 1].x) ** 2 +
                                   (follower.y - self.followers[i - 1].y) ** 2)
                min_dist_timestep = min(min_dist_timestep, dist)

            self.min_distances.append(min_dist_timestep)

            # Log system status
            self.log_system_status(t_step)

        # After simulation, generate the plots
        self.plot_trajectory_snapshots()
        self.plot_velocity_consensus()
        self.plot_min_distances()
        # (Optional) show separate velocity consensus the same way as first script
        # but we already included a velocity consensus plot.

    def log_system_status(self, t_step):
        """
        Log the system status for each vehicle (positions, velocities, queue, etc.)
        """
        logging.info(f"Step {t_step} - System Status:")
        for v in self.vehicles:
            logging.info(f"  Vehicle {v.id}: "
                         f"Position=({v.x:.2f}, {v.y:.2f}), "
                         f"Velocity={v.v_x:.2f}, "
                         f"Queue Size={len(v.message_queue)}, "
                         f"Recovery Mode={v.recovery_mode}")

    ##################################################
    # Plotting (same as second script)
    ##################################################
    def plot_trajectory_snapshots(self):
        plt.figure(figsize=(15, 8))
        plt.suptitle('Vehicle Platooning Trajectory Snapshots', fontsize=14)
        t_samples = [
            int(self.time_steps * 0.01), int(self.time_steps * 0.2),
            int(self.time_steps * 0.4),  int(self.time_steps * 0.6),
            int(self.time_steps * 0.8),  int(self.time_steps * 0.99)
        ]

        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers + 1):
                if i == 0:
                    plt.plot(self.x_history[i][t], self.y_history[i][t], '.r', markersize=10)  # leader in red
                else:
                    plt.plot(self.x_history[i][t], self.y_history[i][t], '.k', markersize=5)

            # Road boundaries (for illustration)
            leader_x = max(self.x_history[0]) if self.x_history[0] else 100
            plt.plot([0, leader_x + 20], [0, 0], 'b-', linewidth=2)
            plt.plot([0, leader_x + 20], [self.road_width, self.road_width], 'b-', linewidth=2)

            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.title(f't={t * self.dt:.2f} sec')
            plt.ylim(-5, self.road_width + 5)
        plt.tight_layout()
        plt.show()

    def plot_velocity_consensus(self):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_followers + 1))
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(self.time_points, self.v_history[i], label=label, color=colors[i], linewidth=2.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity Consensus Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)
        plt.tight_layout()
        plt.show()

    def plot_min_distances(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_points, self.min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.title('Minimum Inter-Vehicle Distance Over Time')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 25)
        plt.tight_layout()
        plt.show()


##################################################
# Main Execution
##################################################
if __name__ == "__main__":
    # You can adjust the number of followers as desired
    num_followers = 10
    simulation = LeaderFollowerSimulation(num_followers)

    # Define (optional) attack parameters to simulate DDoS/flood
    # Attack starts at step=50 and ends at step=70
    attack_params = {
        'start': 50,       # step index
        'end': 150,         # step index
        'targets': [2, 6], # vehicle IDs to attack (follower #1 has id=1, #2 has id=2, etc.)
        'intensity': 500,  # number of messages to flood each step
        'type': 'flood'
    }

    # Run the simulation with attack
    simulation.run_simulation(attack_params)
