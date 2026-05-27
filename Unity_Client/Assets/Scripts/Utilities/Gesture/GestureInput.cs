using System;
using System.Collections.Generic;
using System.Text;

[Serializable]
public class GestureInput : BaseGestureData
{
	public List<List<List<float>>> left_joint_positions;
	public List<List<List<float>>> right_joint_positions;

	public GestureInput(
		string label,
		List<List<List<float>>> leftJointPositions,
		List<List<List<float>>> rightJointPositions,
		List<List<List<float>>> leftJointRotations,
		List<List<List<float>>> rightJointRotations,
		List<List<float>> leftWristPositions,
		List<List<float>> rightWristPositions,
		List<List<float>> leftWristRotations,
		List<List<float>> rightWristRotations
	) : base(label, leftJointRotations, rightJointRotations, leftWristPositions, rightWristPositions, leftWristRotations, rightWristRotations)
	{
		this.left_joint_positions = leftJointPositions;
		this.right_joint_positions = rightJointPositions;

		// Validation
		if (leftJointPositions != null &&
			rightJointPositions != null &&
			leftWristPositions != null)
		{
			if (leftJointPositions.Count != rightJointPositions.Count ||
				rightJointPositions.Count != leftWristPositions.Count)
			{
				throw new InvalidOperationException(
					$"Mismatched frame counts:\n" +
					$"left_joint_positions={leftJointPositions.Count}, " +
					$"right_joint_positions={rightJointPositions.Count}, " +
					$"left_wrist_positions={leftWristPositions.Count}"
				);
			}
		}
		else
		{
			throw new InvalidOperationException("One or more joint/wrist lists were not initialized before validation.");
		}
	}

	public GestureInput DeepCopy()
	{
		// Helper function to copy 3-level nested lists
		List<List<List<float>>> Copy3DList(List<List<List<float>>> source)
		{
			var copy = new List<List<List<float>>>(source.Count);
			foreach (var frame in source)
			{
				var frameCopy = new List<List<float>>(frame.Count);
				foreach (var joint in frame)
				{
					frameCopy.Add(new List<float>(joint)); // copy joint list
				}
				copy.Add(frameCopy);
			}
			return copy;
		}

		// Helper function to copy 2D lists
		List<List<float>> Copy2DList(List<List<float>> source)
		{
			var copy = new List<List<float>>(source.Count);
			foreach (var row in source)
			{
				copy.Add(new List<float>(row));
			}
			return copy;
		}

		return new GestureInput(
			this.label,
			Copy3DList(this.left_joint_positions),
			Copy3DList(this.right_joint_positions),
			Copy3DList(this.left_joint_rotations),
			Copy3DList(this.right_joint_rotations),
			Copy2DList(this.left_wrist_positions),
			Copy2DList(this.right_wrist_positions),
			Copy2DList(this.left_wrist_rotations),
			Copy2DList(this.right_wrist_rotations)
		);
	}

	public override string ToString()
	{
		StringBuilder sb = new StringBuilder();
		sb.AppendLine($"GestureInput: {label}");

		sb.AppendLine("Left Joint Positions:");
		ListUtilities.AppendNestedList(sb, left_joint_positions);

		sb.AppendLine("Right Joint Positions:");
		ListUtilities.AppendNestedList(sb, right_joint_positions);

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
