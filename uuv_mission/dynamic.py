from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        # You are required to implement this method
        """
        Create a Mission from a CSV file.

        Supported formats:
        - CSV with header containing columns for reference, cave_height and cave_depth
            (column names may include keywords like 'reference'/'ref', 'height'/'ceiling',
            'depth'/'floor').
        - CSV without header with three columns in order: reference, cave_height, cave_depth.
        """
        # Try reading with a header first
        data = None
        try:
            data = np.genfromtxt(file_name, delimiter=',', names=True, dtype=float)
        except Exception:
            data = None

        if data is not None and data.dtype.names is not None:
            names = [n.lower() for n in data.dtype.names]

            def find_column(candidates):
                for cand in candidates:
                     for n in names:
                         if cand in n:
                            return n
                return None

            ref_name = find_column(['reference', 'ref'])
            height_name = find_column(['cave_height', 'height', 'ceiling', 'top'])
            depth_name = find_column(['cave_depth', 'depth', 'floor', 'bottom'])

            if ref_name and height_name and depth_name:
                reference = np.asarray(data[ref_name], dtype=float)
                cave_height = np.asarray(data[height_name], dtype=float)
                cave_depth = np.asarray(data[depth_name], dtype=float)
                return cls(reference, cave_height, cave_depth)

        # Fallback: load as plain numeric CSV (no header)
        arr = np.loadtxt(file_name, delimiter=',')
        if arr.ndim == 1:
            if arr.size != 3:
                raise ValueError("CSV must contain three columns (reference, cave_height, cave_depth)")
            reference = np.array([arr[0]])
            cave_height = np.array([arr[1]])
            cave_depth = np.array([arr[2]])
        else:
            if arr.shape[1] < 3:
                raise ValueError("CSV must contain at least three columns (reference, cave_height, cave_depth)")
            reference = arr[:, 0].astype(float)
            cave_height = arr[:, 1].astype(float)
            cave_depth = arr[:, 2].astype(float)

        return cls(reference, cave_height, cave_depth)
    


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()
            # Call your controller here 
            actions[t] = self.controller.update(mission.reference[t], observation_t)
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)

class controller:
    """
    PD controller class for ClosedLoop.
    Usage:
        ctrl = controller(kp=1.0, kd=0.1, dt=1.0, output_limits=(-10,10))
        action = ctrl.update(reference, measurement)
        ctrl.reset()
    """
    def __init__(self, kp: float, kd: float, dt: float = 1.0, output_limits: tuple = None):
        self.kp = float(kp)
        self.kd = float(kd)
        self.dt = float(dt)
        self.prev_error = 0.0
        self.output_limits = output_limits

    def reset(self):
        self.prev_error = 0.0

    def update(self, reference: float, measurement: float) -> float:
        error = float(reference) - float(measurement)
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u = self.kp * error + self.kd * derivative

        if self.output_limits is not None:
            lo, hi = self.output_limits
            if lo is not None:
                u = max(lo, u)
            if hi is not None:
                u = min(hi, u)

        return float(u)
