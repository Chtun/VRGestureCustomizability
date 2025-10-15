using UnityEngine;

public class MainMenuManager : MonoBehaviour
{
	[SerializeField] private GestureHTTPClient gestureHTTPClient;
	[SerializeField] private GestureWebSocketStreamer gestureWebSocketStreamer;

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
}
