import numpy as np
import matplotlib.pyplot as plt
import random
from time import time, sleep
from collections import deque
import logging
from datetime import datetime
import hashlib
import json
# Additional import for creating the final table
import pandas as pd


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
        self.global_anomaly_count = 0  # track total anomalies

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
            if velocity_change > 20:
                anomalies.append(f"Suspicious velocity change: {velocity_change:.2f}")

        # If any anomalies found, increment global counter
        if anomalies:
            self.global_anomaly_count += len(anomalies)

        return anomalies


###############################################################################
# Blockchain Class (with DDoS transaction-rate limiting)
###############################################################################
class Blockchain:
    def __init__(self, num_agents):
        """
        The blockchain tracks transactions (vehicle states).
        It also enforces a transaction rate limit to mitigate DDoS.
        """
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # A signature list for valid vehicle IDs (simple demonstration)
        self.sings = [self.get_sign(i) for i in range(num_agents + 1)]
        self.node_states = {}  # Store the on-chain state history of each node

        # Transaction rate-limiting data
        self.transaction_counts = {}
        self.MAX_TX_PER_BLOCK = 5  # Maximum transactions per node per block
        self.suspicious_nodes = set()

        # Create the genesis block
        self.new_block(previous_hash='1', proof=100)

    def register_node(self, vehicle_id, state):
        """
        Add a new node (vehicle) if it has a valid signature.
        """
        sign = self.get_sign(vehicle_id)
        if vehicle_id in self.nodes:
            logging.info(f"[Blockchain] Vehicle {vehicle_id} already registered.")
        else:
            if sign in self.sings:
                self.nodes.add(vehicle_id)
                self.node_states[vehicle_id] = [state]
                logging.info(f"[Blockchain] Vehicle {vehicle_id} registered successfully.")
            else:
                logging.warning(f"[Blockchain] Registration failed: Invalid signature for {vehicle_id}.")

    def flag_node_as_suspicious(self, vehicle_id):
        """
        Mark a node as suspicious; remove it from the blockchain network.
        """
        self.suspicious_nodes.add(vehicle_id)
        if vehicle_id in self.nodes:
            self.nodes.remove(vehicle_id)
        logging.warning(f"[Blockchain] Node {vehicle_id} flagged and removed from the network.")

    def new_transaction(self, vehicle_id, status):
        """
        Creates a new transaction if within rate limit.
        """
        if vehicle_id not in self.nodes:
            logging.warning(f"[Blockchain] Vehicle {vehicle_id} not recognized.")
            return None

        # If node is already suspicious, reject it
        if vehicle_id in self.suspicious_nodes:
            logging.warning(f"[Blockchain] Vehicle {vehicle_id} blocked (suspicious).")
            return None

        # Check or initialize transaction count
        if vehicle_id not in self.transaction_counts:
            self.transaction_counts[vehicle_id] = 0

        # Rate limit check
        if self.transaction_counts[vehicle_id] >= self.MAX_TX_PER_BLOCK:
            logging.warning(
                f"[DDoS] Vehicle {vehicle_id} exceeded TX limit ({self.MAX_TX_PER_BLOCK})."
            )
            self.flag_node_as_suspicious(vehicle_id)
            return None

        # Otherwise accept transaction
        self.current_transactions.append({
            'vehicle_id': vehicle_id,
            'status': status,
        })
        # Update node state
        if vehicle_id not in self.node_states:
            self.node_states[vehicle_id] = []
        self.node_states[vehicle_id].append(status)

        self.transaction_counts[vehicle_id] += 1
        return self.last_block['index'] + 1

    def new_block(self, proof, previous_hash=None):
        """
        Creates a new block, resets transaction counts.
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.chain.append(block)
        self.current_transactions = []

        # Reset transaction counts for the next block
        self.transaction_counts = {}

        return block

    def check_for_attacks(self, current_step, v_l):
        """
        FDI-like check: if a node’s velocity is too far from consensus, fix or flag it.
        For brevity, only a simple check is shown here.
        """
        if current_step < 10:
            return
        for vehicle_id in list(self.nodes):
            # Trivial approach: if velocity is too far from leader velocity, we "fix" it
            if len(self.node_states[vehicle_id]) > 0:
                current_vx = self.node_states[vehicle_id][-1]['vx']
                if abs(current_vx - v_l) > 0.01:
                    logging.warning(f"[FDI] Potential anomaly for vehicle {vehicle_id}. Resetting velocity.")
                    self.calc_new_velocity(vehicle_id, v_l)

    def calc_new_velocity(self, vehicle_id, consensus_velocity):
        """
        Overwrite the last transaction for the suspicious vehicle with a corrected velocity
        """
        for txn in reversed(self.current_transactions):
            if txn['vehicle_id'] == vehicle_id:
                old_vx = txn['status']['vx']
                txn['status']['vx'] = consensus_velocity
                logging.info(
                    f"[FDI] Velocity of {vehicle_id} changed from {old_vx} to {consensus_velocity}"
                )
                break

    @staticmethod
    def hash(block):
        """
        SHA-256 hash of a block
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, last_block):
        """
        Simple PoW: find a number 'proof' so that hash(last_proof, proof, last_hash) starts with '00'.
        """
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)
        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:2] == "00"

    def get_sign(self, vehicle_id):
        return hashlib.sha256(str(vehicle_id).encode()).hexdigest()[:8]


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
    def __init__(self,blockchain, 
                 id,
                 x=0.0, 
                 y=0.0, 
                 psi=0.0, 
                 vx=10.0, 
                 vy=0.0, 
                 r=0.0, 
                 dt=0.01,
                 ):
        self.id  = id  # For logging or debug
        # States
        self.x   = x
        self.y   = y
        self.psi = psi
        self.vx  = vx
        self.vy  = vy
        self.r   = r
        self.blockchain = blockchain

        # Vehicle parameters
        self.m   = 1500.0    # mass [kg]
        self.L_f = 1.2       # CG to front axle [m]
        self.L_r = 1.6       # CG to rear axle [m]
        self.I_z = 2250.0    # Yaw moment of inertia [kg*m^2]
        self.C_f = 19000.0   # Front cornering stiffness [N/rad]
        self.C_r = 20000.0   # Rear cornering stiffness [N/rad]

        self.dt  = dt  # integration step
        # Register the vehicle as a node in the blockchain
        self.blockchain.register_node(self.id, self.get_state())

        # Message queue for simulating network traffic
        self.message_queue = deque(maxlen=1000)  # limit queue size
        self.processing_delay = 0.001
        self.last_update_time = time()

        # Flags for anomaly and recovery
        self.recovery_mode = False

        # Used to store velocity history for plotting
        self.velocity_history = []


    def update_dynamics(self, security_monitor, simulation):
        """
        - Process messages in the queue within a 0.1s budget
        - Detect anomalies via SecurityMonitor
        - Possibly switch to recovery mode
        - Also update global counters for processed messages
        """
        current_time = time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time

        # Process messages from the queue
        start_time = time()
        while self.message_queue and (time() - start_time) < 0.1:
            msg = self.message_queue.popleft()
            # Count that we've processed one message
            simulation.total_messages_processed += 1
            sleep(self.processing_delay)

        # Update Security Monitor
        queue_size = len(self.message_queue)
        security_monitor.track_message_rate(self.id, queue_size)
        security_monitor.track_velocity_change(self.id, self.vx)

        anomalies = security_monitor.detect_anomalies(self.id)
        if anomalies:
            logging.warning(f"Vehicle {self.id} - Anomalies detected: {anomalies}")
            # print(
            #     f"🚨 [Anomaly Detected] Vehicle ID: {self.id} has anomalies: {', '.join(anomalies)}. "
            #     f"Entering Recovery Mode."
            # )
            self.recovery_mode = True

        # If in recovery mode, reduce speed
        if self.recovery_mode:
            self.apply_recovery_control()

        self.velocity_history.append(self.vx)

    def apply_recovery_control(self):
        """
        Example of a simplistic recovery: reduce velocity if anomalies are detected.
        """
        self.vx *= 0.95  # Gradually slow down if under attack or anomaly

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
        # Finally, broadcast new transaction to the blockchain
        self.broadcast_state()

    def broadcast_state(self):
        """
        Publish (vehicle_id, state) to the blockchain as a transaction.
        """
        state = self.getstate()
        self.blockchain.new_transaction(self.id, state)


    def get_state(self):
        return (self.x, self.y, self.psi, self.vx, self.vy, self.r)

    def getstate(self):
        """
        Get the current state of the vehicle for logging or debugging.
        """
        return {
            'x': round(float(self.x), 2),
            'y': round(float(self.y), 2),
            'yaw': round(float(self.psi), 2),
            'vx': round(float(self.vx), 2),
            'vy': round(float(self.vy), 2)
        }

    def check_and_update_from_blockchain(self, agents):
        """
        After mining, vehicles can reconcile their local states with the final blockchain state
        if desired (optional).
        """
        last_block = self.blockchain.chain[-1]
        for txn in last_block['transactions']:
            veh_id = txn['vehicle_id']
            status = txn['status']
            for ag in agents:
                if ag.id == veh_id:
                    # e.g., unify x, vx from the chain
                    ag.x = status['x']
                    ag.vx = status['vx']

##################################################
# Distributed Attack Simulation (DDoS)
##################################################
def simulate_distributed_attack(
    attackers,
    vehicles,
    target_ids,
    intensity=1000,
    attack_type="flood",
    simulation=None
):
    """
    For each attacker, generate flood traffic for the target vehicles.
    We also increment the simulation's total_messages_sent counter.
    """
    if attack_type != "flood":
        logging.warning(f"Attack type '{attack_type}' not supported.")
        return

    for attacker in attackers:
        for vehicle in vehicles:
            if vehicle.id in target_ids:
                flood_messages = attacker.generate_attack_traffic(intensity)
                # Increment total messages SENT
                if simulation is not None:
                    simulation.total_messages_sent += len(flood_messages)

                # Put them in the vehicle's queue
                vehicle.message_queue.extend(flood_messages)

                logging.info(
                    f"[DDoS] Attacker {attacker.id} -> Vehicle {vehicle.id}: "
                    f"Flooded {intensity} messages. Queue size={len(vehicle.message_queue)}"
                )
                # print(
                #     f"⚠️ [DDoS Attack] Attacker ID:{attacker.id} -> Vehicle ID:{vehicle.id}, "
                #     f"{intensity} messages. QueueSize={len(vehicle.message_queue)}"
                # )

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

        # Lateral Gains: "2-state" feedback on (vy, r)
        # delta = -K_vy*vy - K_r*(r - r_ref)  (plus feedforward, if needed)
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
        # Simple feedback: delta = -K_vy*vy - K_r*(r - r_ref)
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
    def __init__(self, num_followers, dt, blockchain):
        self.dt = dt
        self.blockchain = blockchain
        self.time_steps = int(100.0 / dt)  
        self.num_followers = num_followers
        # Leader (we can choose to use or skip an inner loop here)
        self.leader = DynamicBicycleModel(blockchain=self.blockchain,id=0, x=100, y=10, vx= 10, dt=dt)
        
        # For simplicity, let's do direct control for leader:
        self.leader_speed_ref = 6.0  # We'll keep it constant
        self.kp_leader = 1.0
        self.road_width = 20
        self.ttc_history = []  # Track minimum TTC over time
        # Follower vehicles, each with its own inner controller
        self.followers = []
        self.controllers = []
        for i in range(num_followers):
            veh = DynamicBicycleModel(blockchain=self.blockchain,id=i + 1, x=100 - 14*(i+1),
                                      y=10,
                                      vx=3*(i+1),
                                      dt=dt
                                     )
            self.followers.append(veh)



            ctrl = InnerLoopController(dt=dt)
            self.controllers.append(ctrl)

        self.vehicles = [self.leader] + self.followers  # combined list
        self.elapsed_times = []
        # Create multiple attackers (for distributed attack)
        self.attackers = [
            Attacker(attacker_id=1001),
            Attacker(attacker_id=1002)
        ]
        self.total_attack_delay_times = []
        self.max_delay = 0  # Initialize max delay

        # Outer loop platoon gains
        self.desired_gap = 10.0
        self.k_s   = 0.15   # how strongly we correct spacing error -> vx_ref
        self.k_v   = 0.2   # optionally might do relative speed
        self.k_ey  = 0.6   # lateral offset -> r_ref
        self.k_epsi= 0.6   # heading error -> r_ref

        # Security monitor for all vehicles
        self.security_monitor = SecurityMonitor()

        # == NEW: track message stats to compute packet loss
        self.total_messages_sent = 0
        self.total_messages_processed = 0
        self.min_distances = []
        self.x_history = [[] for _ in range(num_followers + 1)]
        self.y_history = [[] for _ in range(num_followers + 1)]
        self.v_history = [[] for _ in range(num_followers + 1)]


    def calculate_ttc(self):
        ttc_values = []

        # Leader -> first follower
        if self.followers:
            first_follower = self.followers[0]
            rel_vel = first_follower.vx - self.leader.vx
            rel_pos = self.leader.x - first_follower.x 
            if rel_pos > 0 and rel_vel > 0:
                ttc = rel_pos / rel_vel
            else:
                ttc = float('inf')
            ttc_values.append(ttc)

        # Follower -> follower
        for i in range(1, len(self.followers)):
            front = self.followers[i-1]
            rear = self.followers[i]
            rel_vel = rear.vx - front.vx
            rel_pos = front.x - rear.x 
            if rel_pos > 0 and rel_vel > 0:
                ttc = rel_pos / rel_vel
            else:
                ttc = float('inf')
            ttc_values.append(ttc)

        return min(ttc_values) if ttc_values else float('inf')

    def run_simulation(self, attack_params=None):
        print(self.followers)
        # For logging
        time_points = []


        for step in range(self.time_steps):
            logging.info(f"\n--- Simulation Step {step} ---")
            start_time = time()

            t = step * self.dt
            time_points.append(t)

            # If attack parameters exist and we are within the defined range
            if attack_params:
                start_step = attack_params.get('start', 0)
                end_step = attack_params.get('end', 0)
                if start_step <= step <= end_step:
                    simulate_distributed_attack(
                        attackers=self.attackers,
                        vehicles=self.vehicles,
                        target_ids=attack_params['targets'],
                        intensity=attack_params.get('intensity', 500),
                        attack_type=attack_params.get('type', 'flood'),
                        simulation=self  # pass reference for stats
                    )

            end_time = time()
            delay = end_time - start_time
            self.total_attack_delay_times.append(delay)

            # Update max delay if the current delay is greater
            if delay > self.max_delay:
                self.max_delay = delay

            #-----------------------------------------------
            # Leader Control (just do a simple speed P-control)
            #-----------------------------------------------

            a_leader = self.kp_leader*(self.leader_speed_ref - self.leader.vx)
            # keep steering = 0 for leader (straight)
            delta_leader = 0.0
            # Security-based update
            self.leader.update_dynamics(self.security_monitor, simulation=self)
            self.leader.update(a_leader, delta_leader)
            #if t == 0: print("Initial leader state:", self.leader.get_state())
            self.x_history[0].append(self.leader.x)
            self.y_history[0].append(self.leader.y)

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
                # vx_ref = vx0 + k_s*(e_s)    (follower tries to match leader's speed plus correct spacing)
                vx_ref = vx0 + self.k_s * (e_s)+ self.k_v * (vx0 - vx1)

                # r_ref = r0 + k_ey*e_y + k_epsi*e_psi
                # or simpler: just use the lateral error
                r_ref  = r0 + self.k_ey*e_y + self.k_epsi*e_psi

                #--- Now call the follower's inner loop controller
                a_f, delta_f = self.controllers[i].control(
                    vx1, vy1, r1,
                    vx_ref, r_ref
                )
                # Security-based update
                follower.update_dynamics(self.security_monitor, simulation=self)
                #--- Update follower dynamics
                follower.update(a_f, delta_f)

                #--- For logging
                self.x_history[i+1].append(follower.x)
                self.y_history[i+1].append(follower.y)

                # Distance to preceding vehicle (Euclidean)
                dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                if dist < min_dist_timestep:
                    min_dist_timestep = dist

            # Keep track of min distance among all follower pairs for plotting
            self.min_distances.append(min_dist_timestep)
            
            # Calculate and store TTC
            min_ttc = self.calculate_ttc()
            self.ttc_history.append(min_ttc)


            # 5) Blockchain checks & mining each step
            # - check for FDI vs. leader velocity, etc.
            start_time = time()
            self.blockchain.check_for_attacks(step, self.leader.vx)

            proof = self.blockchain.proof_of_work(self.blockchain.last_block)
            self.blockchain.new_block(proof, self.blockchain.hash(self.blockchain.last_block))
            end_time = time()
            # Calculate the time taken for the loop to complete
            elapsed_time = end_time - start_time
            self.elapsed_times.append(elapsed_time)  # Store elapsed time

            # 6) Reconcile states from the chain (optional)
            for i, follower in enumerate(self.followers):
                follower.check_and_update_from_blockchain(self.followers)
                self.v_history[i+1].append(follower.vx)

            # Leader could also do so if you want fully consistent states:
            self.leader.check_and_update_from_blockchain([self.leader])
            self.v_history[0].append(self.leader.vx)

            # Log system status
            self.log_system_status(step)

        #-------------------------------------------------------
        # After simulation, plot results
        #-------------------------------------------------------
        # self.plot_results(time_points, 
        #                   leader_xhist, leader_yhist, leader_vxhist, 
        #                   follower_xhist, follower_yhist, follower_vxhist)
        all_vxhist = self.v_history  
        self.plot_trajectory(self.x_history, self.y_history)
        self.plot_velocity_and_distance(time_points, all_vxhist, self.min_distances)
        self.plot_ttc_over_time(time_points)
        self.plot_elapsed_times()

    def plot_elapsed_times(self):
        plt.figure(figsize=(10,6))
        plt.bar(range(len(self.total_attack_delay_times)), self.total_attack_delay_times, color='blue')
        plt.xlabel('Simulation Step')
        plt.ylabel('Time delay (s)')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


    def log_system_status(self, t_step):
        logging.info(f"Step {t_step} - System Status:")
        # print(f"📊 [System Status] Step {t_step}:")
        for v in self.vehicles:
            logging.info(
                f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
                f"Vx={v.vx:.2f}, QueueSize={len(v.message_queue)}, "
                f"RecoveryMode={v.recovery_mode}"
            )
            # print(f"  Vehicle {v.id}: Pos=({v.x:.2f}, {v.y:.2f}), "
            #     f"Vx={v.vx:.2f}, QueueSize={len(v.message_queue)}, "
            #     f"RecoveryMode={v.recovery_mode}")

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



##################################################
# Main Execution (with Scenario Table)
##################################################
if __name__ == "__main__":
    # Define multiple scenarios (like a Table)
    scenarios = [

        {
            "name": "Severe Flood",
            "attackers": 3,
            "intensity": 1000,
            "start": 50,
            "end": 150
        },

    ]
    results = []

    for scn in scenarios:
        print("\n========================================")
        print(f"Running Scenario: {scn['name']}")
        print("========================================")

        # Create a new simulation
        num_followers = 5
        blockchain = Blockchain(num_followers)
        sim = LeaderFollowerSimulationWithInnerLoop(num_followers , dt=0.01, blockchain= blockchain)

        # Overwrite default attackers
        new_attackers = []
        for i in range(scn["attackers"]):
            new_attackers.append(Attacker(attacker_id=1001 + i))
        sim.attackers = new_attackers

        # Build attack_params if attackers > 0
        if scn["attackers"] > 0:
            attack_params = {
                'start': scn["start"] if scn["start"] else 0,
                'end': scn["end"] if scn["end"] else sim.time_steps - 1,
                'targets': [2, 4],  # arbitrary chosen
                'intensity': scn["intensity"],
                'type': 'flood'
            }
        else:
            attack_params = None

        # Run
        sim.run_simulation(attack_params)

