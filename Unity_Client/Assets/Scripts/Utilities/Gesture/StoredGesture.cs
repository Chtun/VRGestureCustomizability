// ==========================================================
// 🔹 Data classes for request and response
// ==========================================================
using System;
using System.Collections.Generic;
using System.Text;

[Serializable]
public class StoredGesture : BaseGestureData
{
	public List<List<List<float>>> left_hand_vectors;
	public List<List<List<float>>> right_hand_vectors;

	public StoredGesture(
		string label,
		List<List<List<float>>> leftHandVectors,
		List<List<List<float>>> rightHandVectors,
		List<List<List<float>>> leftJointRotations,
		List<List<List<float>>> rightJointRotations,
		List<List<float>> leftWristPositions,
		List<List<float>> rightWristPositions,
		List<List<float>> leftWristRotations,
		List<List<float>> rightWristRotations
	) : base(label, leftJointRotations, rightJointRotations, leftWristPositions, rightWristPositions, leftWristRotations, rightWristRotations)
	{
		this.left_hand_vectors = leftHandVectors;
		this.right_hand_vectors = rightHandVectors;

		// Only perform validation if lists are actually initialized
		if (leftHandVectors != null &&
			rightHandVectors != null &&
			leftWristPositions != null
		)
		{
			if (leftHandVectors.Count != rightHandVectors.Count ||
				rightHandVectors.Count != leftWristPositions.Count)
			{
				throw new InvalidOperationException(
					$"Mismatched frame counts:\n" +
					$"left_hand_vectors={leftHandVectors.Count}, " +
					$"right_hand_vectors={rightHandVectors.Count}, " +
					$"left_wrist_positions={leftWristPositions.Count}"
				);
			}
		}
		else
		{
			throw new InvalidOperationException("One or more joint/wrist lists were not initialized before validation.");
		}
	}

	public override string ToString()
	{
		StringBuilder sb = new StringBuilder();
		sb.AppendLine($"GestureInput: {label}");

		sb.AppendLine("Left Hand Vectors:");
		ListUtilities.AppendNestedList(sb, left_hand_vectors);

		sb.AppendLine("Right Hand Vectors:");
		ListUtilities.AppendNestedList(sb, right_hand_vectors);

		sb.AppendLine("Left Joint Rotations:");
		ListUtilities.AppendNestedList(sb, left_joint_rotations);

		sb.AppendLine("Right Joint Rotations:");
		ListUtilities.AppendNestedList(sb, right_joint_rotations);

		sb.AppendLine("Left Wrist Positions:");
		ListUtilities.Append2DList(sb, left_wrist_positions);

		sb.AppendLine("Right Wrist Positions:");
		ListUtilities.Append2DList(sb, right_wrist_positions);

		sb.AppendLine("Left Wrist Rotations:");
		ListUtilities.Append2DList(sb, left_wrist_rotations);

		sb.AppendLine("Right Wrist Rotations:");
		ListUtilities.Append2DList(sb, right_wrist_rotations);

		return sb.ToString();
	}
}