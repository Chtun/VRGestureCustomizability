def get_hand_joint_list() -> list:
    """
    Grabs the list of hand joint names.
    """
    return [
        "HandThumbTip",
        "HandThumb3",
        "HandThumb2",
        "HandThumb1",
        "HandIndexTip",
        "HandIndex3",
        "HandIndex2",
        "HandIndex1",
        "HandIndex0",
        "HandMiddleTip",
        "HandMiddle3",
        "HandMiddle2",
        "HandMiddle1",
        "HandMiddle0",
        "HandRingTip",
        "HandRing3",
        "HandRing2",
        "HandRing1",
        "HandRing0",
        "HandPinkyTip",
        "HandPinky3",
        "HandPinky2",
        "HandPinky1",
        "HandPinky0",
    ]

def get_connected_indices_list() -> dict:
    """
    For each index, it specifies the index that comes before it in the finger/hand.
    """

    return {
        0: 1,
        1: 2,
        2: 3,
        3: None,
        4: 5,
        5: 6,
        6: 7,
        7: 8,
        8: None,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: None,
        14: 15,
        15: 16,
        16: 17,
        17: 18,
        18: None,
        19: 20,
        20: 21,
        21: 22,
        22: 23,
        23: None
    }


def get_finger_indices_list() -> list[list]:
    """
    Grabs the list of indices for each finger.
    """
    return [
        [0, 1, 2, 3], # Thumb
        [4, 5, 6, 7, 8], # Index
        [9, 10, 11, 12, 13], # Middle
        [14, 15, 16, 17, 18], # Ring
        [19, 20, 21, 22, 23] # Pinky
    ]

def get_hand_joint_dict() -> tuple[dict, dict]:
    """
    Grabs two dictionaries, one that maps hand joint names to its index and one that maps index to hand joint name.

    Returns:
        (dict that maps hand joint names to its index, dict that maps index to hand joint name)
    """

    hand_joint_names = get_hand_joint_list()

    # Map string -> index
    joint_name_to_index = {name: idx for idx, name in enumerate(hand_joint_names)}

    # Map index -> string
    index_to_joint_name = {idx: name for idx, name in enumerate(hand_joint_names)}

    return joint_name_to_index, index_to_joint_name