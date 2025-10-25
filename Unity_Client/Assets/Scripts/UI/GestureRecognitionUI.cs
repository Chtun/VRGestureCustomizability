using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class GestureRecognitionUI : MonoBehaviour
{
	[Header("References")]
	[SerializeField] private GameObject gesturePrefab; // Prefab containing Image + TMP Text
	[SerializeField] private Transform gestureParent;  // Parent to hold instantiated prefabs
	[SerializeField] private Color detectedColor = Color.green;
	[SerializeField] private Color notDetectedColor = Color.red;
	[SerializeField] private float cooldownDuration = 1f; // seconds

	// Internal dictionary mapping gesture keys to prefab instances
	private Dictionary<string, GestureUIItem> gestureUIItems = new Dictionary<string, GestureUIItem>();

	// Call this at start with the initial dictionary to create all UI elements
	public void InitializeGestures(Dictionary<string, (float distance, bool matched)> gestureData)
	{
		foreach (var kvp in gestureData)
		{
			string gestureKey = kvp.Key;

			if (!gestureUIItems.ContainsKey(gestureKey))
			{
				// Instantiate prefab
				GameObject instance = Instantiate(gesturePrefab, gestureParent);
				instance.name = $"Gesture_{gestureKey}";

				// Get components
				TextMeshProUGUI gestureDistanceText = instance.transform.Find("GestureDistanceText").GetComponent<TextMeshProUGUI>();
				TextMeshProUGUI gestureDetectionText = instance.transform.Find("GestureDetectionText").GetComponent<TextMeshProUGUI>();
				Image detectionImage = instance.GetComponentInChildren<Image>();

				// Store in dictionary
				gestureUIItems[gestureKey] = new GestureUIItem
				{
					GestureDistanceText = gestureDistanceText,
					GestureDetectionText = gestureDetectionText,
					DetectionImage = detectionImage,
					LastUpdateTime = -Mathf.Infinity
				};
			}
		}

		// Update values immediately after creation
		UpdateGestures(gestureData);
	}

	// Call this whenever you want to update distance/matched values
	public void UpdateGestures(Dictionary<string, (float distance, bool matched)> gestureData)
	{
		float currentTime = Time.time;

		foreach (var kvp in gestureData)
		{
			string gestureKey = kvp.Key;
			float distance = kvp.Value.distance;
			bool matched = kvp.Value.matched;

			if (gestureUIItems.TryGetValue(gestureKey, out var uiItem))
			{
				// If the gesture is in cooldown, skip updates until time passes
				if (currentTime - uiItem.LastUpdateTime < cooldownDuration)
					continue;

				// If matched, trigger cooldown
				if (matched)
					uiItem.LastUpdateTime = currentTime;

				if (uiItem.GestureDistanceText != null)
					uiItem.GestureDistanceText.text = $"{gestureKey} Distance: {distance:F2}";

				if (uiItem.GestureDetectionText != null)
					uiItem.GestureDetectionText.text = $"{gestureKey} Detected?";

				if (uiItem.DetectionImage != null)
					uiItem.DetectionImage.color = matched ? detectedColor : notDetectedColor;
			}
		}
	}

	// Internal helper class to store references for each gesture
	private class GestureUIItem
	{
		public TextMeshProUGUI GestureDistanceText;
		public TextMeshProUGUI GestureDetectionText;
		public Image DetectionImage;
		public float LastUpdateTime;
	}
}
