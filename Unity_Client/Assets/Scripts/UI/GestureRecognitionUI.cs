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

	private bool isUIInitialized = false;
	public bool IsUIInitialized => isUIInitialized;

	// Internal dictionary mapping gesture keys to prefab instances
	private Dictionary<string, GestureUIItem> gestureUIItems = new Dictionary<string, GestureUIItem>();

	// Call this at start with the initial dictionary to create all UI elements
	public void InitializeGestures(Dictionary<string, (float distance, bool matched)> gestureData)
	{
		isUIInitialized = true;

		foreach (var kvp in gestureData)
		{
			string gestureKey = kvp.Key;

			if (!gestureUIItems.ContainsKey(gestureKey))
			{
				// Instantiate prefab
				GameObject instance = Instantiate(gesturePrefab, gestureParent);
				instance.name = $"Gesture_{gestureKey}";

				// Get components
				TextMeshProUGUI gestureDistanceText = FindChildComponentByName<TextMeshProUGUI>(instance.transform, "GestureDistanceText");
				TextMeshProUGUI gestureDetectionText = FindChildComponentByName<TextMeshProUGUI>(instance.transform, "GestureDetectionText");

				if (gestureDistanceText == null)
					Debug.LogWarning("GestureDistanceText not found!");
				if (gestureDetectionText == null)
					Debug.LogWarning("GestureDetectionText not found!");

				// Image somewhere in the children
				Image detectionImage = instance.GetComponentInChildren<Image>(true);
				if (detectionImage == null)
					Debug.LogWarning("Image not found!");

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

	/// <summary>
	/// Recursively searches all children for a component of type T on a GameObject with a given name.
	/// </summary>
	private T FindChildComponentByName<T>(Transform parent, string childName) where T : Component
	{
		foreach (Transform child in parent)
		{
			if (child.name == childName)
			{
				T component = child.GetComponent<T>();
				if (component != null)
					return component;
			}

			// Recursively search this child's children
			T found = FindChildComponentByName<T>(child, childName);
			if (found != null)
				return found;
		}

		return null;
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
