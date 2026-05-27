using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class JointDataUI : MonoBehaviour
{
	[Header("References")]
	[SerializeField] public TextMeshProUGUI timestampText;              // UI Text to display time stamp
	[SerializeField] public float updateInterval = 0.1f;     // Seconds between UI updates


	[SerializeField] private Color activeColor = Color.green;
	[SerializeField] private Color inactiveColor = Color.red;

	[Header("Hand Reliability Indicators")]
	[SerializeField] private Image leftHandIndicator;
	[SerializeField] private Image rightHandIndicator;

	[Header("Is Recording Indicator")]
	[SerializeField] private Image isRecordingIndicator;

	private string scriptName = "JointDataUI";

	private void Start()
	{
		if (GestureSystemManager.instance != null)
			StartCoroutine(UpdateUIRoutine());
		else
			Debug.LogError($"[{scriptName}] GestureSystemManager instance not found!");
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
		if (GestureSystemManager.instance == null)
			return;

		// Check if hand data is reliable
		bool rightHandReliable = GestureSystemManager.instance.IsDataReliable(isRightHand: true);
		bool leftHandReliable = GestureSystemManager.instance.IsDataReliable(isRightHand: false);

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
		{
			if (GestureSystemManager.instance.IsRecording())
				isRecordingIndicator.color = activeColor;
			else
			{
				isRecordingIndicator.color = inactiveColor;
			}
		}

	}

}