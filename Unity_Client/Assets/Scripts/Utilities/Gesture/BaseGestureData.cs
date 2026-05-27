using System;
using System.Collections.Generic;

[Serializable]
public abstract class BaseGestureData
{
	public string label;
	public List<List<List<float>>> left_joint_rotations;
	public List<List<List<float>>> right_joint_rotations;
	public List<List<float>> left_wrist_positions;
	public List<List<float>> right_wrist_positions;
	public List<List<float>> left_wrist_rotations;
	public List<List<float>> right_wrist_rotations;

	public BaseGestureData(
		string label,
		List<List<List<float>>> leftJointRotations,
		List<List<List<float>>> rightJointRotations,
		List<List<float>> leftWristPositions,
		List<List<float>> rightWristPositions,
		List<List<float>> leftWristRotations,
		List<List<float>> rightWristRotations
	)
	{

		this.label = label;
		this.left_joint_rotations = leftJointRotations;
		this.right_joint_rotations = rightJointRotations;
		this.left_wrist_positions = leftWristPositions;
		this.right_wrist_positions = rightWristPositions;
		this.left_wrist_rotations = leftWristRotations;
		this.right_wrist_rotations = rightWristRotations;

		// Only perform validation if lists are actually initialized
		if (left_joint_rotations != null &&
			right_joint_rotations != null &&
			left_wrist_positions != null &&
			right_wrist_positions != null &&
			left_wrist_rotations != null &&
			right_wrist_rotations != null)
		{
			if (left_joint_rotations.Count != right_joint_rotations.Count ||
				right_joint_rotations.Count != left_wrist_positions.Count ||
				left_wrist_positions.Count != right_wrist_positions.Count ||
				right_wrist_positions.Count != left_wrist_rotations.Count ||
				left_wrist_rotations.Count != right_wrist_rotations.Count)
			{
				throw new InvalidOperationException(
					$"Mismatched frame counts:\n" +
					$"left_joint_rotations={left_joint_rotations.Count}, " +
					$"right_joint_rotations={right_joint_rotations.Count}, " +
					$"left_wrist_positions={left_wrist_positions.Count}, " +
					$"right_wrist_positions={right_wrist_positions.Count}, " +
					$"left_wrist_rotations={left_wrist_rotations.Count}, " +
					$"right_wrist_rotations={right_wrist_rotations.Count}"
				);
			}
		}
		else
		{
			throw new InvalidOperationException("One or more joint/wrist lists were not initialized before validation.");
		}
	}

}
