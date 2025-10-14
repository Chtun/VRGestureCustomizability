using Oculus.Interaction.Input; // For Hand, HandJointId, Pose
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class JointDataUI : MonoBehaviour
{
	[Header("References")]
	[SerializeField] public JointDataGather jointDataGather;
	[SerializeField] public TextMeshProUGUI jointDataText;              // UI Text to display joint info
	[SerializeField] public float updateInterval = 0.1f;     // Seconds between UI updates

	private void Start()
	{
		if (jointDataGather != null)
			StartCoroutine(UpdateUIRoutine());
		else
			Debug.LogError("JointDataGather reference not set in JointDataUI.");
	}

	private IEnumerator UpdateUIRoutine()
	{
		while (true)
		{
			UpdateJointDataUI();
			yield return new WaitForSeconds(updateInterval);
		}
	}

	private void UpdateJointDataUI()
	{
		if (jointDataGather == null || jointDataText == null)
			return;

		// Collect joint data from your existing script
		Dictionary<HandJointId, Pose> leftJoints = jointDataGather.GetJointData(isRightHand: false);
		Dictionary<HandJointId, Pose> rightJoints = jointDataGather.GetJointData(isRightHand: true);

		Pose leftRoot = jointDataGather.GetRootPose(isRightHand: false);
		Pose rightRoot = jointDataGather.GetRootPose(isRightHand: true);

		// Build display string
		System.Text.StringBuilder sb = new System.Text.StringBuilder();
		string time_str = Time.time.ToString();
		sb.AppendLine($"Time Stamp: {time_str}");
		sb.AppendLine("");

		sb.AppendLine("=== Left Hand ===");
		foreach (var kvp in leftJoints)
		{
			sb.AppendLine($"{kvp.Key}: Pos({kvp.Value.position.x:F2}, {kvp.Value.position.y:F2}, {kvp.Value.position.z:F2}) " +
						  $"Rot({kvp.Value.rotation.x:F2}, {kvp.Value.rotation.y:F2}, {kvp.Value.rotation.z:F2}, {kvp.Value.rotation.w:F2})");
		}
		sb.AppendLine($"Left Root: Pos({leftRoot.position.x:F2}, {leftRoot.position.y:F2}, {leftRoot.position.z:F2}) " +
					  $"Rot({leftRoot.rotation.x:F2}, {leftRoot.rotation.y:F2}, {leftRoot.rotation.z:F2}, {leftRoot.rotation.w:F2})");

		sb.AppendLine("=== Right Hand ===");
		foreach (var kvp in rightJoints)
		{
			sb.AppendLine($"{kvp.Key}: Pos({kvp.Value.position.x:F2}, {kvp.Value.position.y:F2}, {kvp.Value.position.z:F2}) " +
						  $"Rot({kvp.Value.rotation.x:F2}, {kvp.Value.rotation.y:F2}, {kvp.Value.rotation.z:F2}, {kvp.Value.rotation.w:F2})");
		}
		sb.AppendLine($"Right Root: Pos({rightRoot.position.x:F2}, {rightRoot.position.y:F2}, {rightRoot.position.z:F2}) " +
					  $"Rot({rightRoot.rotation.x:F2}, {rightRoot.rotation.y:F2}, {rightRoot.rotation.z:F2}, {rightRoot.rotation.w:F2})");

		// Update UI Text
		jointDataText.text = sb.ToString();
	}
}