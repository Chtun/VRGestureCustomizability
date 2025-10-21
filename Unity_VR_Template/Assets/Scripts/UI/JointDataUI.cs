using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class JointDataUI : MonoBehaviour
{
	[Header("References")]
	[SerializeField] public JointDataGather jointDataGather;
	[SerializeField] public TextMeshProUGUI timestampText;              // UI Text to display time stamp
	[SerializeField] public float updateInterval = 0.1f;     // Seconds between UI updates


	[SerializeField] private Color activeColor = Color.green;
	[SerializeField] private Color inactiveColor = Color.red;

	[Header("Hand Reliability Indicators")]
	[SerializeField] private Image leftHandIndicator;
	[SerializeField] private Image rightHandIndicator;

	[Header("Is Recording Indicator")]
	[SerializeField] private Image isRecordingIndicator;

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
		if (jointDataGather == null)
			return;

		// Check if hand data is reliable
		bool rightHandReliable = this.jointDataGather.IsDataReliable(isRightHand: true);
		bool leftHandReliable = this.jointDataGather.IsDataReliable(isRightHand: false);

		// Update circle colors
		if (leftHandIndicator != null)
			leftHandIndicator.color = leftHandReliable ? activeColor : inactiveColor;

		if (rightHandIndicator != null)
			rightHandIndicator.color = rightHandReliable ? activeColor : inactiveColor;

		// Update timestamp text
		if (timestampText != null)
		{
			System.Text.StringBuilder sb = new System.Text.StringBuilder();
			string time_str = Time.time.ToString("F2");
			sb.AppendLine($"Time Stamp: {time_str}");

			timestampText.text = sb.ToString();
		}


		if (isRecordingIndicator != null)
			isRecordingIndicator.color = activeColor;
	}

}