"""Package reference initialization for GestureBuilder reimplementation."""
from .model import *
from .utilities import *

__version__ = "0.1.0"
__author__ = "Chitsein Htun"
__all__ = [
    # Model
    'VQVAE',


    # Utilities
    'cosine_denoised_distance',
    'convert_joint_to_hand_vector',

    'HandDataset',

    'get_hand_joint_list',
    'get_connected_indices_list',
    'get_finger_indices_list',
    'get_hand_joint_dict',
]