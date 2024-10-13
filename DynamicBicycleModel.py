import numpy as np
import matplotlib.pyplot as plt

class DynamicBicycleModel:


    def __init__(self, id, x=0.0, y=0.0, psi=0.0, vx=10.0, vy=0.0, r=0.0, delta=0.0, ax=0.0):
        self.id = id  # Vehicle ID
        self.x = x  # X position
        self.y = y  # Y position
        self.psi = psi  # Yaw angle
        self.v_x = vx  # Longitudinal velocity
        self.v_y = vy  # Lateral velocity
        self.r = r  # Yaw rate
        self.delta = delta  # Steering angle (control input)
        self.ax = ax # Longitudinal acceleration (driven by throttle or braking).

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

        k1 = self.f(xx , a , delta)
        k2 = self.f(xx + self.dt /2 * k1 , a , delta)
        k3 = self.f(xx + self.dt /2 * k2 , a , delta)
        k4 = self.f(xx + self.dt * k3 , a , delta)

        xx = xx + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        self.x = xx[0]  # Global x position
        self.y = xx[1]  # Global y position
        self.psi = xx[2]  # Yaw angle
        self.v_x = xx[3]  # Longitudinal velocity (initial value)
        self.v_y = xx[4]  # Lateral velocity
        self.r = xx[5]  # Yaw rate


    def f(self, xx , a, delta):
        """
        calculating the states's drivatives using Runge-Kutta Algorithm
        """
        
        x = xx[0]  # Global x position
        y = xx[1]  # Global y position
        psi = xx[2]  # Yaw angle
        v_x = xx[3]  # Longitudinal velocity (initial value)
        v_y = xx[4]  # Lateral velocity
        r = xx[5]  # Yaw rate
        
        # Calculate slip angles
        alpha_f = delta - np.arctan2((v_y + self.L_f * r), v_x)
        alpha_r = -np.arctan2((v_y - self.L_r * r), v_x)

        # Calculate lateral forces
        F_yf = self.C_f * alpha_f
        F_yr = self.C_r * alpha_r

        # State variables
        x_dot = v_x * np.cos(psi) - v_y * np.sin(psi) # Global x position
        y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)  # Global y position
        psi_dot= r  # Yaw angle
        v_x_dot= a - (F_yf * np.sin(delta)) / self.m + v_y * r  # Longitudinal velocity (initial value)
        v_y_dot= ((F_yf * np.cos(delta) + F_yr) / self.m - v_x * r)  # Lateral velocity
        r_dot= (self.L_f * F_yf* np.cos(delta)  - self.L_r * F_yr) / self.I_z  # Yaw rate
        
        return np.array([x_dot, y_dot, psi_dot, v_x_dot, v_y_dot, r_dot])
    

    def get_state(self):
        """
        Get the current state of the vehicle
        :return: A dictionary containing the vehicle's state
        """
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'V_x': self.v_x,
            'V_y': self.v_y,
            'yaw': self.psi,
            'Yaw Rate': self.r
        }    

    def get_velocity(self):
        """
        Calculate and return the total velocity of the vehicle.
        :return: The magnitude of the total velocity.
        """
        return np.sqrt(self.vx**2 + self.vy**2)

# Create an instance of the DynamicBicycleModel class
vehicle = DynamicBicycleModel(1)

# Simulation loop
time_steps = int(10000)
x_history = []
y_history = []
vx_history = []
vy_history = []
delta_his = []
timesteps = []

for t in range(time_steps):
    # Sinusoidal steering input (delta) with time
    delta = 0.1 * np.sin(0.2 * t * vehicle.dt)  # Steering angle (radians)
    delta_his.append(delta)

    a = 0  # Constant acceleration (m/s^2)
    
    # Update vehicle state
    vehicle.update(a, delta)
    
    # Store position for plotting
    x_history.append(vehicle.x)
    y_history.append(vehicle.y)
    vx_history.append(vehicle.v_x)
    vy_history.append(vehicle.v_y)

    timesteps.append(t * vehicle.dt)  # Time in seconds


# Plot vehicle trajectory
plt.figure(figsize=(10, 6))
plt.plot(timesteps, delta_his, label='Steering Angle (delta)')
plt.xlabel('Time (s)')
plt.ylabel('Control Inputs')
plt.title('Steering Angle and Longitudinal Acceleration Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trajectory
plt.figure()
plt.plot(x_history, y_history, label='Vehicle Path')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Dynamic Bicycle Model Trajectory with Sinusoidal Steering Input')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

#print(vy_history)
plt.figure()
plt.plot(timesteps, vx_history, label='vx (Longitudinal Velocity)')
plt.plot(timesteps, vy_history, label='vy (Lateral Velocity)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Longitudinal and Lateral Velocities Over Time')
plt.legend()
plt.grid(True)
plt.show()

