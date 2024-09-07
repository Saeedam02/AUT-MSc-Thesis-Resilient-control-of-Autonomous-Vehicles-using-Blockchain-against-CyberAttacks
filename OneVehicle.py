import numpy as np
import matplotlib.pyplot as plt

#Vehicle dynamic model (bicycle model)
class Vehicle:
    def __init__(self, id, x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=0.0, r=0.0, delta=0.0, ax=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.yaw = yaw  # Yaw angle
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.r = r  # Yaw rate
        self.delta = delta  # Steering angle (control input)
        self.ax = ax # Longitudinal acceleration (driven by throttle or braking).

        # Vehicle-specific parameters
        self.mass = 1500  # kg
        self.lf = 1.2  # Distance from CG to front axle (m)
        self.lr = 1.6  # Distance from CG to rear axle (m)
        self.Iz = 2250  # Yaw moment of inertia (kg*m^2)
        self.Cf = 19000  # Cornering stiffness front (N/rad)
        self.Cr = 20000  # Cornering stiffness rear (N/rad)
        self.dt = 0.01  # Time step (s)

    def update_dynamics(self):
        """
        Update the vehicle's state based on the bicycle model dynamics
        """
        # Calculate slip angles (clamped to avoid excessive forces)
        #alpha_f = np.clip(self.delta - (self.vy + self.lf * self.r) / self.vx, -0.1, 0.1)
        #alpha_r = np.clip(-(self.vy - self.lr * self.r) / self.vx, -0.1, 0.1)

        alpha_f = self.delta - (self.vy + self.lf * self.r) / self.vx
        alpha_r = - (self.vy - self.lr * self.r) / self.vx


        Fyf = self.Cf * alpha_f  # Lateral force at the front tire
        Fyr = self.Cr * alpha_r  # Lateral force at the rear tire

        # Calculate state derivatives

        #In reality, lateral velocity is limited by tire friction and road conditions. 
        #damping_factor = 0.7  # Adjust this value for appropriate damping
        vy_dot = ((Fyf + Fyr) / self.mass - self.vx * self.r) #- damping_factor * self.vy
        vx_dot = self.ax - (self.r * self.vy)
        r_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        x_dot = self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)
        y_dot = self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)
        yaw_dot = (self.vx / (self.lf + self.lr)) * np.tan(self.delta)

        # Longitudinal velocity affected by drag to keep it realistic
        #drag_force = 0.01 * self.vx ** 2  # Drag proportional to the square of vx
        self.vx += vx_dot * self.dt

        # Update the state
        self.vy += vy_dot * self.dt
        self.r += r_dot * self.dt
        self.yaw += yaw_dot * self.dt
        self.x += x_dot * self.dt
        self.y += y_dot * self.dt

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw
        }

    def get_velocity(self):
        """
        Calculate and return the total velocity of the vehicle.
        :return: The magnitude of the total velocity.
        """
        return np.sqrt(self.vx**2 + self.vy**2)
    

vehicle = Vehicle(1)
vehicle.x = 2
vehicle.y = 3
#vehicle.delta = np.deg2rad(1)

vehicle.vx = 10  # Initial velocity (m/s)
vehicle.vy = 0 # Initial lateral velocity (m/s)
vehicle.yaw = 0  # Initial yaw angle (rad)
vehicle.r = 0  # Initial yaw rate (rad/s)


x_histories = []
y_histories = []
vx_history = []
vy_history = []

# Time and velocity storage
time_steps = []
velocities = []
delta_his = []

for t in range(10000):
    vehicle.delta = np.deg2rad(np.sin(t * 0.001) * 2)  # Varying steering angle 
    vehicle.update_dynamics()
    delta_his.append(vehicle.delta)
    velocities.append(vehicle.get_velocity())
    x_histories.append(vehicle.x)
    y_histories.append(vehicle.y)
    vx_history.append(vehicle.vx)
    vy_history.append(vehicle.vy)


    time_steps.append(t * vehicle.dt)  # Time in seconds

#print(vx_history)

# Plot the velocities over time
plt.figure(figsize=(10, 6))
plt.plot(time_steps, delta_his, label='delta')
plt.xlabel('Time (s)')
plt.ylabel('delta (rad)')
plt.title('delta over time')
plt.legend()
plt.grid(True)
plt.show()

#print(vy_history)
plt.figure(figsize=(10, 6))
plt.plot(time_steps, vx_history, label='vx (Longitudinal Velocity)')
plt.plot(time_steps, vy_history, label='vy (Lateral Velocity)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Longitudinal and Lateral Velocities Over Time')
plt.legend()
plt.grid(True)
plt.show()


# Plot the trajectories of the vehicle
plt.figure(figsize=(10, 8))
plt.plot(x_histories, y_histories, label=f'Vehicle')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectories')
plt.legend()
plt.grid(True)
plt.show()

# Plot the velocities over time
plt.figure(figsize=(10, 6))
plt.plot(time_steps, velocities, label='Vehicle Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Vehicle Velocity Over Time')
plt.legend()
plt.grid(True)
plt.show()