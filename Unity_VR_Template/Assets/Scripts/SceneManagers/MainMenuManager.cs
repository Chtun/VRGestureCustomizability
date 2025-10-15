using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using TMPro;
using UnityEngine;

public class MainMenuManager : MonoBehaviour
{
	[SerializeField] private GestureHTTPClient gestureHTTPClient;
	[SerializeField] private GestureWebSocketStreamer gestureWebSocketStreamer;
	[SerializeField] private TMP_Text gestureStatusText; // UI Text to show status

	private bool matchFound = false;
	private Coroutine resetMatchCoroutine;

	void Awake()
	{
		// Auto-find the HTTP client if not assigned in Inspector
		if (gestureHTTPClient == null)
			gestureHTTPClient = FindFirstObjectByType<GestureHTTPClient>();

		if (gestureHTTPClient == null)
			Debug.LogError("[MainMenuManager] GestureHTTPClient not found in the scene!");

		// Auto-find the Web Socket Streamer if not assigned in Inspector
		if (gestureWebSocketStreamer == null)
			gestureWebSocketStreamer = FindFirstObjectByType<GestureWebSocketStreamer>();

		if (gestureWebSocketStreamer == null)
			Debug.LogError("[MainMenuManager] GestureWebSocketStreamer not found in the scene!");

		// Subscribe to WebSocket messages
		if (gestureWebSocketStreamer != null)
			gestureWebSocketStreamer.OnGestureDataReceived += HandleGestureMessage;
	}

	void Start()
	{
		if (gestureHTTPClient != null)
		{
			// Fetch gestures from server at startup
			StartCoroutine(gestureHTTPClient.GetGestures(OnGesturesFetched));
		}
		else
		{
			Debug.LogWarning("Gestures were not grabbed at the start due to the client not being found.");
		}

		// Initialize text
		UpdateGestureStatusText();
	}

	private void OnGesturesFetched(string jsonResponse)
	{
		if (!string.IsNullOrEmpty(jsonResponse))
		{
			Debug.Log($"[MainMenuManager] Gestures loaded.");
		}
		else
		{
			Debug.LogError("[MainMenuManager] Failed to load gestures from server.");
		}
	}

	private void HandleGestureMessage(string json)
	{
		try
		{
			var obj = JObject.Parse(json);
			bool match = obj.Value<bool>("match");
			string label = obj.Value<string>("label");
			float distance = obj.Value<float>("dtw_distance");

			if (match)
			{
				matchFound = true;
				UpdateGestureStatusText(label, distance);

				// Restart the 2-second reset timer if already running
				if (resetMatchCoroutine != null)
					StopCoroutine(resetMatchCoroutine);

				resetMatchCoroutine = StartCoroutine(ResetMatchAfterDelay(2f));
			}
			else
			{
				// If no match, do not reset matchFound (it remains false if already false)
				if (!matchFound)
					UpdateGestureStatusText(null, distance);
			}
		}
		catch (Exception ex)
		{
			Debug.LogError($"Failed to parse gesture JSON: {ex}");
		}
	}

	private IEnumerator ResetMatchAfterDelay(float delay)
	{
		yield return new WaitForSeconds(delay);
		matchFound = false;
		UpdateGestureStatusText(null);
	}

	private void UpdateGestureStatusText(string label = null, float distance = -1)
	{
		if (gestureStatusText == null)
			return;

		if (matchFound && !string.IsNullOrEmpty(label))
			gestureStatusText.text = $"Gesture match found: {label}, distance to it is: {distance}.";
		else
			gestureStatusText.text = $"No gesture match, distance to closest is: {distance}.";
	}
}
