// ==========================================================
// 🔹 Data classes for request and response
// ==========================================================
using System;
using System.Collections.Generic;
using System.Text;

[Serializable]
public class GestureInput
{
	public string label;
	public List<List<List<float>>> left_joints;
	public List<List<List<float>>> right_joints;
	public List<List<float>> left_wrist;
	public List<List<float>> right_wrist;

	public GestureInput(
		string label,
		List<List<List<float>>> leftJoints,
		List<List<List<float>>> rightJoints,
		List<List<float>> leftWrist,
		List<List<float>> rightWrist
	)
	{
		this.label = label;
		this.left_joints = leftJoints;
		this.right_joints = rightJoints;
		this.left_wrist = leftWrist;
		this.right_wrist = rightWrist;
	}

	public override string ToString()
	{
		StringBuilder sb = new StringBuilder();
		sb.AppendLine($"GestureInput: {label}");

		sb.AppendLine("Left Joints:");
		AppendNestedList(sb, left_joints);

		sb.AppendLine("Right Joints:");
		AppendNestedList(sb, right_joints);

		sb.AppendLine("Left Wrist:");
		Append2DList(sb, left_wrist);

		sb.AppendLine("Right Wrist:");
		Append2DList(sb, right_wrist);

		return sb.ToString();
	}

	private void AppendNestedList(StringBuilder sb, List<List<List<float>>> data)
	{
		foreach (var frame in data)
		{
			sb.Append("[ ");
			foreach (var joint in frame)
			{
				sb.Append("[");
				sb.Append(string.Join(", ", joint));
				sb.Append("] ");
			}
			sb.AppendLine("]");
		}
	}

	private void Append2DList(StringBuilder sb, List<List<float>> data)
	{
		foreach (var frame in data)
		{
			sb.AppendLine("[ " + string.Join(", ", frame) + " ]");
		}
	}


}