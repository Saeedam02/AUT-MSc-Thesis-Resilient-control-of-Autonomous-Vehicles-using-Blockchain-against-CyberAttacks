import numpy as np
import matplotlib.pyplot as plt
import random
from time import time, sleep
from collections import deque
import logging
from datetime import datetime

##################################################
# Logging Configuration
##################################################
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(
    f'platooning_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Stream (Console) Handler
stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)

# Add Handlers to the Logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

##################################################
# Security Monitor (for queue-based DDoS detection)
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

        # Check for high message rate anomalies
        if len(self.message_rates.get(vehicle_id, [])) >= 2:
            rate = np.mean(list(self.message_rates[vehicle_id]))
            # Example threshold for high message rate
            if rate > 100:
                anomalies.append(f"High message rate: {rate:.2f}")

        # Check for sudden velocity change anomalies
        if len(self.velocity_changes.get(vehicle_id, [])) >= 2:
            velocities = list(self.velocity_changes[vehicle_id])
            velocity_change = abs(velocities[-1] - velocities[-2])
            # Example threshold for sudden velocity change
            if velocity_change > 20:
                anomalies.append(f"Suspicious velocity change: {velocity_change:.2f}")

        return anomalies

##################################################
# Attacker Class (multiple sources for DDoS)
##################################################
class Attacker:
    """
    An attacker with a specific ID. Each attacker can generate flood messages
    to perform the attack from multiple sources.
    """
    def __init__(self, attacker_id):
        self.id = attacker_id

    def generate_attack_traffic(self, intensity):
        """
        Generate a large number of messages (intensity) for flooding.
        For identification, include attacker_id in the message.
        """
        flood_messages = [
            {'timestamp': time(), 'type': 'status_update', 'attacker_id': self.id}
            for _ in range(intensity)
        ]
        return flood_messages

##################################################
# Dynamic Bicycle Model + Security Additions
##################################################
class DynamicBicycleModel:
    """
    Dynamic bicycle model with message queue and security monitor
    to simulate DoS/DDoS attacks.
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
        self.message_queue = deque(maxlen=1000)  # Limit queue size
        self.processing_delay = 0.001
        self.last_update_time = time()

        # Flags for anomaly and recovery
        self.recovery_mode = False

        # Vehicle-specific parameters
        self.m = 1500
        self.L_f = 1.2
        self.L_r = 1.6
        self.I_z = 2250
        self.C_f = 19000
        self.C_r = 20000
        self.dt = 0.01

        # Used to store velocity history for plotting
        self.velocity_history = []

    def update_dynamics(self, security_monitor):
        """
        - Process message queue within a time budget (0.1 seconds)
        - Detect anomalies (high message rate or sudden velocity changes)
        - Apply recovery mode if anomalies are detected
        """
        current_time = time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process messages in the queue (within 0.1 second budget)
        start_time = time()
        messages_processed = 0
        while self.message_queue and (time() - start_time) < 0.1:
            msg = self.message_queue.popleft()
            # If desired, you can read msg['attacker_id'] here for tracking
            messages_processed += 1
            sleep(self.processing_delay)

        # Report message rate and velocity changes to SecurityMonitor
        queue_size = len(self.message_queue)
        security_monitor.track_message_rate(self.id, queue_size)
        security_monitor.track_velocity_change(self.id, self.v_x)

        # Check for anomalies
        anomalies = security_monitor.detect_anomalies(self.id)
        if anomalies:
            logging.warning(f"Vehicle {self.id} - Anomalies detected: {anomalies}")
            # **Print Statement for Anomaly Detection**
            print(
                f"🚨 [Anomaly Detected] Vehicle ID: {self.id} has detected anomalies: {', '.join(anomalies)}. "
                f"Entering Recovery Mode."
            )
            self.recovery_mode = True

        # If in recovery mode, reduce speed
        if self.recovery_mode:
            self.apply_recovery_control()

        # Record velocity for plotting
        self.velocity_history.append(self.v_x)

    def apply_recovery_control(self):
        """
        Simplest recovery strategy: reduce speed by 5%
        """
        self.v_x *= 0.95

    ############################
    # Remaining dynamic model code
    ############################
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

        self.x = xx[0]
        self.y = xx[1]
        self.psi = xx[2]
        self.v_x = xx[3]
        self.v_y = xx[4]
        self.r = xx[5]

    def f(self, xx, a, delta):
        """
        State derivatives (part of the bicycle model).
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
        General vehicle information
        """
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.v_x), 2),
            'vy': round(float(self.v_y), 2)
        }

##################################################
# Distributed Attack Simulation (DDoS)
##################################################
def simulate_distributed_attack(
    attackers,       # List of attackers (from Attacker class)
    vehicles,        # List of all vehicles (Leader + Followers)
    target_ids,      # IDs of victim vehicles
    intensity=1000,
    attack_type="flood"
):
    """
    In this function, each attacker individually sends flood messages to the target vehicles.
    """
    if attack_type != "flood":
        logging.warning(f"Attack type '{attack_type}' not yet supported.")
        return

    # Iterate over each attacker
    for attacker in attackers:
        # Iterate over each target vehicle
        for vehicle in vehicles:
            if vehicle.id in target_ids:
                # Generate flood messages by the current attacker
                flood_messages = attacker.generate_attack_traffic(intensity)
                # Append messages to the target vehicle's message queue
                vehicle.message_queue.extend(flood_messages)
                logging.info(
                    f"[DDoS] Attacker {attacker.id} -> Vehicle {vehicle.id}: "
                    f"Flooded {intensity} messages. Queue size = {len(vehicle.message_queue)}"
                )
                # **Print Statement for Terminal Output**
                print(
                    f"⚠️  [DDoS Attack] Attacker ID: {attacker.id} is attacking "
                    f"Vehicle ID: {vehicle.id} with {intensity} flood messages. "
                    f"New Queue Size: {len(vehicle.message_queue)}"
                )

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
        self.vehicles = [self.leader] + self.followers

        # Create multiple attackers (for distributed attack)
        self.attackers = [
            Attacker(attacker_id=1001),
            Attacker(attacker_id=1002)
        ]

        self.num_followers = num_followers
        self.desired_gap = 10
        self.dt = 0.05
        self.time_steps = int(50 / self.dt)
        self.road_width = 20

        # For plotting
        self.x_history = [[] for _ in range(num_followers + 1)]
        self.y_history = [[] for _ in range(num_followers + 1)]
        self.v_history = [[] for _ in range(num_followers + 1)]
        self.min_distances = []
        self.time_points = np.arange(0, self.time_steps) * self.dt
        self.ttc_history = []  # Track minimum TTC over time
        self.total_attack_delay_times = []
        # Security monitor shared by all vehicles
        self.security_monitor = SecurityMonitor()

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


    def run_simulation(self, attack_params=None):
        """
        Main simulation loop with optional DDoS/flood attack.
        """
        for t_step in range(self.time_steps):
            logging.info(f"\n--- Simulation Step {t_step} ---")

            start_time = time()

            # If attack parameters exist and we are within the defined range
            if attack_params:
                start_step = attack_params.get('start', 0)
                end_step = attack_params.get('end', 0)
                if start_step <= t_step <= end_step:
                    # Use the new distributed attack simulation function
                    simulate_distributed_attack(
                        attackers=self.attackers,
                        vehicles=self.vehicles,
                        target_ids=attack_params['targets'],
                        intensity=attack_params.get('intensity', 500),
                        attack_type=attack_params.get('type', 'flood')
                    )
            end_time = time()
            total_attack_delay_time = end_time - start_time
            self.total_attack_delay_times.append(total_attack_delay_time)

            # 1) Update Leader
            v_target = 6.0
            k_p = 1
            a_l = k_p * (v_target - self.leader.v_x)
            # Security-based update
            self.leader.update_dynamics(self.security_monitor)
            # Physical model update
            self.leader.update(a_l, 0)

            # 2) Record Leader's state
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)
            self.v_history[0].append(self.leader.v_x)

            min_dist_timestep = float('inf')

            # 3) Update Followers
            for i, follower in enumerate(self.followers):
                distance_to_leader = self.leader.x - follower.x - self.desired_gap * (i + 1)
                a_f = 1.0 * distance_to_leader  # Simple P-control

                follower.update_dynamics(self.security_monitor)
                follower.update(a_f, 0)

                # History
                self.x_history[i + 1].append(follower.x)
                self.y_history[i + 1].append(follower.y)
                self.v_history[i + 1].append(follower.v_x)

                # Distance between follower and the one in front
                if i == 0:
                    dist = np.sqrt((follower.x - self.leader.x) ** 2 + (follower.y - self.leader.y) ** 2)
                else:
                    dist = np.sqrt((follower.x - self.followers[i - 1].x) ** 2 +
                                   (follower.y - self.followers[i - 1].y) ** 2)
                min_dist_timestep = min(min_dist_timestep, dist)

            self.min_distances.append(min_dist_timestep)
            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)
            # Log system status
            self.log_system_status(t_step)

        # After the loop, plot the graphs
        self.plot_trajectory_snapshots()
        self.plot_velocity_and_min_distance()
        self.plot_elapsed_times()
        self.plot_ttc_over_time()

    def log_system_status(self, t_step):
        logging.info(f"Step {t_step} - System Status:")
        print(f"📊 [System Status] Step {t_step}:")
        for v in self.vehicles:
            logging.info(
                f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                f"Vx={v.v_x:.2f}, QueueSize={len(v.message_queue)}, "
                f"RecoveryMode={v.recovery_mode}"
            )
            print(f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                f"Vx={v.v_x:.2f}, QueueSize={len(v.message_queue)}, "
                f"RecoveryMode={v.recovery_mode}")

    ##################################################
    # Plotting
    ##################################################
    def plot_trajectory_snapshots(self):
        plt.figure(figsize=(15, 8))
        t_samples = [
            int(self.time_steps * 0.01), int(self.time_steps * 0.2),
            int(self.time_steps * 0.4),  int(self.time_steps * 0.6),
            int(self.time_steps * 0.8),  int(self.time_steps * 0.99)
        ]

        for idx, t in enumerate(t_samples):
            plt.subplot(3, 2, idx + 1)
            for i in range(self.num_followers + 1):
                if i == 0:
                    plt.plot(self.x_history[i][t], self.y_history[i][t], '.r', markersize=10)  # Leader in red
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

    def plot_velocity_and_min_distance(self):
        # Create a figure with two subplots
        plt.figure(figsize=(10, 12))

        # Subplot for Velocity Consensus
        plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
        colors = ['red'] + ['black', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']  # Custom colors
        for i in range(self.num_followers + 1):
            label = 'Leader' if i == 0 else f'Follower {i}'
            plt.plot(self.time_points, self.v_history[i], label=label, color=colors[i], linewidth=2.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.ylim(0, 15)  # Adjusted range to see velocity differences

        # Subplot for Minimum Distance
        plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
        plt.plot(self.time_points, self.min_distances, 'b-', label='Minimum Distance')
        plt.axhline(y=self.desired_gap, color='r', linestyle='--', label='Desired Gap')
        plt.xlabel('Time [s]')
        plt.ylabel('Distance [m]')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_elapsed_times(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.total_attack_delay_times)), self.total_attack_delay_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Time delay(seconds)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_ttc_over_time(self):
        """
        Plot the Time to Collision (TTC) over time.
        """
        time_points = np.arange(len(self.ttc_history)) * self.dt  # Generate correct time points
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, self.ttc_history, 'g-', label='Minimum TTC')
        plt.axhline(y=1, color='r', linestyle='--', label='TTC Threshold (1s)')
        plt.axhline(y=2, color='orange', linestyle='--', label='TTC Threshold (2s)')
        plt.xlabel('Time [s]')
        plt.ylabel('TTC [s]')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, time_points[-1])
        plt.tight_layout()
        plt.show()


##################################################
# Main Execution
##################################################
if __name__ == "__main__":
    num_followers = 10
    simulation = LeaderFollowerSimulation(num_followers)


    attack_params = {
        'start': 50,       # Start time step for the attack
        'end': 150,        # End time step for the attack
        'targets': [2, 6], # IDs of vehicles to attack
        'intensity': 10000,  # Number of flood messages per attacker
        'type': 'flood'
    }

    simulation.run_simulation(attack_params)
