using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

public class GesturePracticeManager : MonoBehaviour
{
	[SerializeField] private GestureHTTPClient gestureHTTPClient;
	[SerializeField] private GestureWebSocketStreamer gestureWebSocketStreamer;
	[SerializeField] private GestureRecognitionUI gestureRecognitionUI;

	private bool isUIInitialized = false;

	void Awake()
	{
		// Auto-find clients if not assigned
		if (gestureHTTPClient == null)
			gestureHTTPClient = FindFirstObjectByType<GestureHTTPClient>();
		if (gestureHTTPClient == null)
			Debug.LogError("[GesturePracticeManager] GestureHTTPClient not found!");

		if (gestureWebSocketStreamer == null)
			gestureWebSocketStreamer = FindFirstObjectByType<GestureWebSocketStreamer>();
		if (gestureWebSocketStreamer == null)
			Debug.LogError("[GesturePracticeManager] GestureWebSocketStreamer not found!");

		if (gestureRecognitionUI == null)
			gestureRecognitionUI = FindFirstObjectByType<GestureRecognitionUI>();
		if (gestureRecognitionUI == null)
			Debug.LogError("[GesturePracticeManager] GestureRecognitionUI not found!");

		if (gestureWebSocketStreamer != null)
			gestureWebSocketStreamer.OnGestureDataReceived += HandleGestureMessage;
	}

	void Start()
	{
		if (gestureHTTPClient != null)
			StartCoroutine(gestureHTTPClient.GetGestures(OnGesturesFetched));
	}

	private void OnGesturesFetched(string jsonResponse)
	{
		if (!string.IsNullOrEmpty(jsonResponse))
			Debug.Log("[GesturePracticeManager] Gestures loaded.");
		else
			Debug.LogError("[GesturePracticeManager] Failed to load gestures.");
	}

	private void HandleGestureMessage(string json)
	{
		try
		{
			var obj = JObject.Parse(json);

			// The backend now sends: {"gesture_results": {"wave": [0.23, true], "fist": [0.87, false], ...}}
			var resultsToken = obj["gesture_results"];
			if (resultsToken == null)
			{
				Debug.LogWarning("No gesture_results found in message.");
				return;
			}

			// Convert to a dictionary of string -> (float distance, bool matched)
			var gestureDict = new Dictionary<string, (float distance, bool matched)>();
			foreach (var prop in resultsToken.Children<JProperty>())
			{
				string key = prop.Name;
				JArray valueArray = prop.Value as JArray;
				if (valueArray != null && valueArray.Count == 2)
				{
					float distance = valueArray[0].Value<float>();
					bool matched = valueArray[1].Value<bool>();
					gestureDict[key] = (distance, matched);
				}
			}

			// Initialize the UI if not done yet
			if (!isUIInitialized)
			{
				gestureRecognitionUI.InitializeGestures(gestureDict);
				isUIInitialized = true;
			}
			else
			{
				gestureRecognitionUI.UpdateGestures(gestureDict);
			}
		}
		catch (Exception ex)
		{
			Debug.LogError($"Failed to parse gesture JSON: {ex}");
		}
	}
}
