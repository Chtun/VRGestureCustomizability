// ==========================================================
// 🔹 Data classes for request and response
// ==========================================================
using System;
using System.Collections.Generic;

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
}