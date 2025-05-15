"""
Utility functions for relativistic neural network operations.

This module provides helper functions and utilities for implementing 
relativistic concepts in neural networks, particularly those inspired
by the Terrell-Penrose effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union, Dict, Any
import math
import numpy as np


def lorentz_factor(velocity: Tensor) -> Tensor:
    """
    Calculate the Lorentz factor (gamma) for a given velocity.
    
    The Lorentz factor is a key quantity in special relativity that appears
    in various relativistic effects like time dilation and length contraction.
    
    Args:
        velocity (Tensor): Velocity as a fraction of the speed of light (c)
                           Either a scalar or vector.
                           
    Returns:
        Tensor: Lorentz factor γ = 1/√(1-v²)
    """
    # Handle scalar or vector velocity
    if velocity.dim() == 0:
        v_squared = velocity ** 2
    else:
        v_squared = torch.sum(velocity ** 2, dim=-1)
    
    # Ensure v_squared is less than 1 (speed of light)
    v_squared = torch.clamp(v_squared, 0.0, 0.999)
    
    # Calculate gamma
    gamma = 1.0 / torch.sqrt(1.0 - v_squared)
    
    return gamma


def lorentz_contraction(length: Tensor, velocity: Tensor) -> Tensor:
    """
    Calculate the Lorentz-contracted length of an object.
    
    In special relativity, the length of an object moving relative to an observer
    appears contracted along the direction of motion.
    
    Args:
        length (Tensor): Proper length of the object (length in its rest frame)
        velocity (Tensor): Velocity as a fraction of light speed
        
    Returns:
        Tensor: Contracted length
    """
    gamma = lorentz_factor(velocity)
    contracted_length = length / gamma
    return contracted_length


def time_dilation(time: Tensor, velocity: Tensor) -> Tensor:
    """
    Calculate the dilated time due to relativistic effects.
    
    In special relativity, time appears to pass more slowly for observers in motion
    relative to another reference frame.
    
    Args:
        time (Tensor): Proper time (time in the rest frame)
        velocity (Tensor): Velocity as a fraction of light speed
        
    Returns:
        Tensor: Dilated time
    """
    gamma = lorentz_factor(velocity)
    dilated_time = time * gamma
    return dilated_time


def velocity_addition(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Compute the relativistic addition of velocities.
    
    The relativistic velocity addition formula differs from the classical vector
    addition and ensures that velocities don't exceed the speed of light.
    
    Args:
        v1 (Tensor): First velocity (fraction of c)
        v2 (Tensor): Second velocity (fraction of c)
        
    Returns:
        Tensor: Resultant velocity according to relativistic addition
    """
    # Handle 1D (directional) velocities
    if v1.dim() == 1 and v2.dim() == 1:
        # Calculate magnitudes
        v1_mag = torch.norm(v1)
        v2_mag = torch.norm(v2)
        
        # Get unit vectors
        v1_unit = v1 / (v1_mag + 1e-8)
        v2_unit = v2 / (v2_mag + 1e-8)
        
        # Special case: parallel velocities
        if torch.allclose(v1_unit, v2_unit, atol=1e-6) or torch.allclose(v1_unit, -v2_unit, atol=1e-6):
            # Use scalar formula for parallel velocities
            dot_product = torch.sum(v1_unit * v2_unit)
            signed_v2_mag = v2_mag * dot_product
            v_resultant_mag = (v1_mag + signed_v2_mag) / (1 + v1_mag * signed_v2_mag)
            
            # Determine direction
            resultant_dir = v1_unit if dot_product > 0 else -v1_unit
            return v_resultant_mag * resultant_dir
        
        # General case: 3D velocity addition
        # Decompose v2 into parallel and perpendicular components relative to v1
        v2_parallel = torch.sum(v2 * v1_unit) * v1_unit
        v2_perp = v2 - v2_parallel
        
        # Apply relativistic velocity addition to parallel component
        gamma1 = lorentz_factor(v1_mag)
        v_parallel_resultant = (v2_parallel + v1) / (1 + torch.sum(v2_parallel * v1) / (v1_mag + 1e-8))
        
        # Transform perpendicular component
        v_perp_resultant = v2_perp / gamma1
        
        # Combine components
        return v_parallel_resultant + v_perp_resultant
    
    # Handle scalar velocities
    else:
        # Simple relativistic velocity addition formula
        return (v1 + v2) / (1 + v1 * v2)


def relativistic_doppler_factor(velocity: Tensor, observer_angle: Optional[Tensor] = None) -> Tensor:
    """
    Calculate the relativistic Doppler factor.
    
    The Doppler factor determines how frequencies are shifted for an observer
    in relative motion. In the Terrell-Penrose effect, different parts of a moving
    object experience different Doppler shifts, affecting their appearance.
    
    Args:
        velocity (Tensor): Velocity as a fraction of light speed
        observer_angle (Tensor, optional): Angle between velocity vector and 
                                           line of sight (radians). 
                                           Defaults to None (assumes head-on observation).
        
    Returns:
        Tensor: Doppler factor
    """
    gamma = lorentz_factor(velocity)
    
    if observer_angle is None:
        # Assume head-on observation
        return gamma * (1 + velocity)  # Approaching observer
    else:
        # General case
        cos_angle = torch.cos(observer_angle)
        return gamma * (1 - velocity * cos_angle)


def terrell_rotation_angle(velocity: Tensor) -> Tensor:
    """
    Calculate the apparent rotation angle from the Terrell-Penrose effect.
    
    In the Terrell-Penrose effect, a rapidly moving object appears rotated
    rather than contracted. This function calculates the apparent rotation angle.
    
    Args:
        velocity (Tensor): Velocity as a fraction of light speed
        
    Returns:
        Tensor: Approximate rotation angle in radians
    """
    velocity = torch.abs(velocity)
    
    # Simple approximation formula for the apparent rotation
    # This is a simplified version; the exact effect is more complex
    angle = torch.atan(velocity / torch.sqrt(1.0 - velocity**2))
    
    return angle


def lorentz_transform_spacetime(coordinates: Tensor, velocity: Tensor) -> Tensor:
    """
    Apply Lorentz transformation to spacetime coordinates.
    
    The Lorentz transformation describes how spacetime coordinates transform
    between reference frames in relative motion.
    
    Args:
        coordinates (Tensor): Spacetime coordinates [batch, ..., 4]
                             where the last dimension is (t, x, y, z)
        velocity (Tensor): Relative velocity as a 3D vector [vx, vy, vz]
                          representing fraction of light speed
        
    Returns:
        Tensor: Transformed spacetime coordinates
    """
    batch_size = coordinates.shape[0]
    orig_shape = coordinates.shape
    
    # Reshape to [batch, -1, 4] for processing
    coords_flat = coordinates.reshape(batch_size, -1, 4)
    
    # Split into time and space components
    t = coords_flat[..., 0]
    x = coords_flat[..., 1:4]
    
    # Calculate relativistic parameters
    v_magnitude = torch.norm(velocity)
    v_magnitude = torch.clamp(v_magnitude, 0.0, 0.999)  # Ensure v < c
    
    gamma = lorentz_factor(v_magnitude)
    
    # Normalize velocity
    v_normalized = velocity / (v_magnitude + 1e-8)
    
    # Calculate dot product between position and velocity
    x_dot_v = torch.sum(x * v_normalized, dim=-1)
    
    # Lorentz transformation
    # t' = γ(t - v·x)
    t_prime = gamma * (t - v_magnitude * x_dot_v)
    
    # x' = x + (γ-1)(v·x)v/v² - γvt
    # First term: perpendicular component unchanged
    # Second term: parallel component transformation
    # Third term: time contribution
    x_prime = x + \
              (gamma - 1) * x_dot_v.unsqueeze(-1) * v_normalized / (v_magnitude**2 + 1e-8) - \
              gamma * t.unsqueeze(-1) * velocity
    
    # Combine transformed coordinates
    transformed = torch.cat([t_prime.unsqueeze(-1), x_prime], dim=-1)
    
    # Reshape back to original dimensions
    return transformed.reshape(orig_shape)


def create_rotation_matrix_from_vectors(v1: Tensor, v2: Tensor) -> Tensor:
    """
    Create a 3D rotation matrix that rotates v1 to align with v2.
    
    This is useful for implementing relativistic transformations where the
    rotation axis depends on the relative orientation of objects.
    
    Args:
        v1 (Tensor): Source vector [3]
        v2 (Tensor): Target vector [3]
        
    Returns:
        Tensor: 3x3 rotation matrix
    """
    # Normalize input vectors
    v1_norm = v1 / (torch.norm(v1) + 1e-8)
    v2_norm = v2 / (torch.norm(v2) + 1e-8)
    
    # Check if vectors are parallel
    if torch.allclose(v1_norm, v2_norm, atol=1e-6):
        # No rotation needed (identity)
        return torch.eye(3, device=v1.device)
    elif torch.allclose(v1_norm, -v2_norm, atol=1e-6):
        # 180 degree rotation around any perpendicular axis
        # Find a perpendicular vector
        if torch.abs(v1_norm[0]) < torch.abs(v1_norm[1]):
            perp = torch.Tensor([1, 0, 0]).to(v1.device)
        else:
            perp = torch.Tensor([0, 1, 0]).to(v1.device)
        
        axis = torch.cross(v1_norm, perp)
        axis = axis / (torch.norm(axis) + 1e-8)
        
        # Create rotation matrix for 180 degrees around axis
        return rotation_matrix_from_axis_angle(axis, math.pi)
    
    # General case: Rodrigues' rotation formula
    # 1. Find rotation axis (cross product)
    axis = torch.cross(v1_norm, v2_norm)
    axis = axis / (torch.norm(axis) + 1e-8)
    
    # 2. Find rotation angle (dot product)
    cos_angle = torch.dot(v1_norm, v2_norm)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # 3. Create rotation matrix
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """
    Create a rotation matrix from an axis and angle.
    
    Args:
        axis (Tensor): Rotation axis (normalized)
        angle (Tensor): Rotation angle in radians
        
    Returns:
        Tensor: 3x3 rotation matrix
    """
    # Rodrigues' rotation formula
    K = torch.zeros(3, 3, device=axis.device)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    
    I = torch.eye(3, device=axis.device)
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    
    return R


class SphericalHarmonics:
    """
    Compute spherical harmonics for relativistic feature transformations.
    
    Spherical harmonics are useful for representing functions on the sphere
    and can be used to implement relativistic transformations efficiently.
    
    Args:
        max_degree (int): Maximum degree of spherical harmonics
        device (torch.device, optional): Device to use. Defaults to None.
    """
    
    def __init__(self, max_degree: int, device: Optional[torch.device] = None):
        self.max_degree = max_degree
        self.device = device
    
    def evaluate(self, theta: Tensor, phi: Tensor) -> Dict[Tuple[int, int], Tensor]:
        """
        Evaluate spherical harmonics at given angles.
        
        Args:
            theta (Tensor): Polar angle in radians [0, π]
            phi (Tensor): Azimuthal angle in radians [0, 2π]
            
        Returns:
            Dict[Tuple[int, int], Tensor]: Dictionary mapping (l, m) to 
                                         spherical harmonic values Y_l^m(θ, φ)
        """
        result = {}
        
        # Iterate over degrees and orders
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                y_lm = self._eval_single(l, m, theta, phi)
                result[(l, m)] = y_lm
        
        return result
    
    def _eval_single(self, l: int, m: int, theta: Tensor, phi: Tensor) -> Tensor:
        """
        Evaluate a single spherical harmonic Y_l^m(θ, φ).
        
        Args:
            l (int): Degree
            m (int): Order
            theta (Tensor): Polar angle
            phi (Tensor): Azimuthal angle
            
        Returns:
            Tensor: Values of Y_l^m(θ, φ)
        """
        # Calculate the associated Legendre polynomial P_l^m(cos θ)
        x = torch.cos(theta)
        p_lm = self._associated_legendre(l, abs(m), x)
        
        # Normalization factor
        # Factor of (-1)^m for Condon-Shortley phase
        if m < 0:
            # Y_l^(-m) = (-1)^m × conj(Y_l^m)
            phase = (-1) ** m
            # For real spherical harmonics, we use sin(m×φ) for m < 0
            phi_part = torch.sin(abs(m) * phi)
        else:
            phase = 1.0
            # For real spherical harmonics, we use cos(m×φ) for m ≥ 0
            phi_part = torch.cos(m * phi)
        
        # Normalization constant (using real spherical harmonics)
        if m == 0:
            norm = math.sqrt((2 * l + 1) / (4 * math.pi))
        else:
            norm = math.sqrt((2 * l + 1) * math.factorial(l - abs(m)) / 
                           (2 * math.pi * math.factorial(l + abs(m))))
            norm *= math.sqrt(2)  # For real spherical harmonics
        
        # Combine to get the spherical harmonic
        y_lm = norm * phase * p_lm * phi_part
        
        return y_lm
    
    def _associated_legendre(self, l: int, m: int, x: Tensor) -> Tensor:
        """
        Compute the associated Legendre polynomial P_l^m(x).
        
        Args:
            l (int): Degree
            m (int): Order
            x (Tensor): Input values (cos θ)
            
        Returns:
            Tensor: P_l^m(x)
        """
        # We compute P_l^m recursively
        # Base cases
        if l == 0 and m == 0:
            return torch.ones_like(x)
        
        if m > l:
            return torch.zeros_like(x)
        
        # For m = l
        if m == l:
            # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^(m/2)
            factor = 1.0
            for i in range(1, m + 1):
                factor *= (2 * i - 1)
            
            result = factor * torch.pow(1.0 - x * x, m / 2.0)
            if m % 2 == 1:
                result = -result
            
            return result
        
        # For m = l-1
        if m == l - 1:
            # P_(m+1)^m(x) = x * (2m+1) * P_m^m(x)
            return x * (2 * m + 1) * self._associated_legendre(m, m, x)
        
        # Recursion relation for l > m+1
        # P_l^m(x) = ((2l-1) * x * P_(l-1)^m(x) - (l+m-1) * P_(l-2)^m(x)) / (l-m)
        term1 = (2 * l - 1) * x * self._associated_legendre(l - 1, m, x)
        term2 = (l + m - 1) * self._associated_legendre(l - 2, m, x)
        
        return (term1 - term2) / (l - m)


class LeviCivitaTensor:
    """
    Levi-Civita tensor implementation for relativistic operations.
    
    The Levi-Civita tensor is useful for implementing cross products and
    rotational transformations in neural network operations.
    
    Args:
        dim (int): Dimensionality of the tensor (3 or 4)
        device (torch.device, optional): Device to use. Defaults to None.
    """
    
    def __init__(self, dim: int = 3, device: Optional[torch.device] = None):
        self.dim = dim
        self.device = device
        self._tensor = self._create_tensor()
    
    def _create_tensor(self) -> Tensor:
        """
        Create the Levi-Civita tensor.
        
        Returns:
            Tensor: Levi-Civita tensor of shape [dim, dim, dim] or [dim, dim, dim, dim]
        """
        if self.dim == 3:
            # Create 3D Levi-Civita tensor (εijk)
            result = torch.zeros(3, 3, 3, device=self.device)
            
            # Set non-zero elements
            # ε_123 = ε_231 = ε_312 = 1
            # ε_132 = ε_321 = ε_213 = -1
            result[0, 1, 2] = 1
            result[1, 2, 0] = 1
            result[2, 0, 1] = 1
            result[0, 2, 1] = -1
            result[2, 1, 0] = -1
            result[1, 0, 2] = -1
            
            return result
        
        elif self.dim == 4:
            # Create 4D Levi-Civita tensor (εijkl)
            result = torch.zeros(4, 4, 4, 4, device=self.device)
            
            # For 4D, we use the determinant definition
            # εijkl = determinant of the matrix [ei, ej, ek, el]
            # where ei is the i-th standard basis vector
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        for l in range(4):
                            # Skip if any indices are equal (εijkl = 0 if any two indices are equal)
                            if i == j or i == k or i == l or j == k or j == l or k == l:
                                continue
                            
                            # Create permutation and get sign
                            perm = [i, j, k, l]
                            sign = 1
                            
                            # Calculate sign by counting inversions
                            for m in range(4):
                                for n in range(m + 1, 4):
                                    if perm[m] > perm[n]:
                                        sign *= -1
                            
                            result[i, j, k, l] = sign
            
            return result
        
        else:
            raise ValueError(f"Dimension {self.dim} not supported. Use 3 or 4.")
    
    def __call__(self, *indices: int) -> Tensor:
        """
        Get the value of the Levi-Civita tensor for specific indices.
        
        Args:
            *indices: Indices to access
            
        Returns:
            Tensor: Value at the specified indices
        """
        return self._tensor[indices]
    
    def contract(self, tensor: Tensor, dim1: int, dim2: int) -> Tensor:
        """
        Contract the Levi-Civita tensor with another tensor.
        
        This operation is useful for implementing cross products and curl operations.
        
        Args:
            tensor (Tensor): Tensor to contract with
            dim1 (int): First dimension to contract
            dim2 (int): Second dimension to contract
            
        Returns:
            Tensor: Contracted tensor
        """
        # Ensure input tensor is on the same device
        if tensor.device != self._tensor.device:
            tensor = tensor.to(self._tensor.device)
        
        if self.dim == 3:
            # For 3D Levi-Civita, implement cross product
            if tensor.dim() == 1 and tensor.shape[0] == 3:
                # Cross product with a vector
                result = torch.zeros(3, device=self._tensor.device)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            result[i] += self._tensor[i, j, k] * tensor[j]
                
                return result
            
            elif tensor.dim() == 2 and tensor.shape[0] == 3 and tensor.shape[1] == 3:
                # Contract with a matrix
                result = torch.zeros_like(tensor)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                result[i, j] += self._tensor[i, k, l] * tensor[k, l]
                
                return result
            
            else:
                raise ValueError(f"Unsupported tensor shape for 3D contraction: {tensor.shape}")
        
        elif self.dim == 4:
            # 4D contractions are more complex and depend on the specific use case
            # Implement as needed for relativistic transformations
            raise NotImplementedError("4D Levi-Civita contractions not yet implemented")


class MinkowskiMetric:
    """
    Minkowski metric for relativistic spacetime operations.
    
    The Minkowski metric is fundamental in special relativity for calculating
    spacetime intervals and implementing Lorentz transformations.
    
    Args:
        signature (str, optional): Metric signature. Defaults to "mostly_minus".
                                   Options: "mostly_minus" (-,+,+,+) or "mostly_plus" (+,-,-,-).
        device (torch.device, optional): Device to use. Defaults to None.
    """
    
    def __init__(self, signature: str = "mostly_minus", device: Optional[torch.device] = None):
        self.signature = signature
        self.device = device
        self._metric = self._create_metric()
    
    def _create_metric(self) -> Tensor:
        """
        Create the Minkowski metric tensor.
        
        Returns:
            Tensor: 4x4 Minkowski metric tensor
        """
        if self.signature == "mostly_minus":
            # (-,+,+,+) signature: g_μν = diag(-1, 1, 1, 1)
            return torch.diag(torch.tensor([-1., 1., 1., 1.], device=self.device))
        elif self.signature == "mostly_plus":
            # (+,-,-,-) signature: g_μν = diag(1, -1, -1, -1)
            return torch.diag(torch.tensor([1., -1., -1., -1.], device=self.device))
        else:
            raise ValueError(f"Unknown signature: {self.signature}")
    
    def interval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Calculate the spacetime interval between two events.
        
        The spacetime interval is invariant under Lorentz transformations.
        
        Args:
            x (Tensor): First spacetime point [batch, 4]
            y (Tensor): Second spacetime point [batch, 4]
            
        Returns:
            Tensor: Spacetime interval ds² = gμν(xμ-yμ)(xν-yν)
        """
        # Calculate displacement vector
        delta = x - y
        
        # Apply metric to calculate interval
        # ds² = gμν dx^μ dx^ν
        result = torch.einsum('bi,ij,bj->b', delta, self._metric, delta)
        
        return result
    
    def raise_index(self, vector: Tensor) -> Tensor:
        """
        Raise the index of a 4-vector (convert from covariant to contravariant).
        
        Args:
            vector (Tensor): Covariant 4-vector [batch, 4]
            
        Returns:
            Tensor: Contravariant 4-vector [batch, 4]
        """
        return torch.einsum('bi,ij->bj', vector, self._metric)
    
    def lower_index(self, vector: Tensor) -> Tensor:
        """
        Lower the index of a 4-vector (convert from contravariant to covariant).
        
        Args:
            vector (Tensor): Contravariant 4-vector [batch, 4]
            
        Returns:
            Tensor: Covariant 4-vector [batch, 4]
        """
        # For Minkowski metric, raising and lowering are similar
        # with sign changes based on metric signature
        return torch.einsum('bi,ij->bj', vector, self._metric)
