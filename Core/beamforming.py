"""
Spectra - Beamforming Algorithms
Implements Conventional and MVDR beamforming.
"""

import numpy as np


class Beamformer:
    def __init__(self, array):
        self.array = array

    def conventional_weights(self, steering_angle):
        return self.array.steering_vector(steering_angle)

    def mvdr_weights(self, steering_angle, R):
        """
        steering_angle : desired signal direction (deg)
        R              : covariance matrix (NxN)
        """
        a = self.array.steering_vector(steering_angle).reshape(-1, 1)
        R_inv = np.linalg.inv(R)
        numerator = R_inv @ a
        denominator = (a.conj().T @ R_inv @ a)
        w = numerator / denominator
        return w.flatten()

    def null_steering_weights(self, desired_angle, interference_angles):
        """
        Creates weights that null interference directions.
        """
        A = []
        A.append(self.array.steering_vector(desired_angle))

        for angle in interference_angles:
            A.append(self.array.steering_vector(angle))

        A = np.column_stack(A)
        b = np.zeros(A.shape[1])
        b[0] = 1  # desired signal

        w = np.linalg.pinv(A) @ b
        return w
