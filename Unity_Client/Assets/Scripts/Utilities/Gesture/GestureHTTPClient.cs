using Newtonsoft.Json;
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Handles REST API communication with the Python FastAPI gesture server.
/// </summary>
public class GestureHTTPClient : MonoBehaviour
{
	[Header("API Settings (auto-loaded from config)")]
	private string baseUrl = "http://127.0.0.1:8000";

	private void Awake()
	{
		Config config = Config.LoadConfig();
		baseUrl = config.GetHTTPURL();
		Debug.Log($"Loaded server URL from config: {baseUrl}");
	}

	[Serializable]
	public class AddGestureResponse
	{
		public int status_code;
		public string message;
	}

	// ==========================================================
	// 🔹 POST /add_gesture/
	// ==========================================================
	public IEnumerator AddGesture(GestureInput gesture, Action<bool> onComplete = null)
	{
		string json = JsonConvert.SerializeObject(gesture);
		byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(json);

		using (UnityWebRequest www = new UnityWebRequest($"{baseUrl}/add_gesture", "POST"))
		{
			www.uploadHandler = new UploadHandlerRaw(bodyRaw);
			www.downloadHandler = new DownloadHandlerBuffer();
			www.SetRequestHeader("Content-Type", "application/json");

			yield return www.SendWebRequest();

			if (www.result == UnityWebRequest.Result.Success)
			{
				// Parse server response
				AddGestureResponse response = JsonConvert.DeserializeObject<AddGestureResponse>(www.downloadHandler.text);

				bool success = response.status_code == 0; // 0 = OK
				Debug.Log($"AddGesture response: {response.message} (Success: {success})");

				onComplete?.Invoke(true);
			}
			else
			{
				Debug.LogError($"Error adding gesture: {www.error}\n{www.downloadHandler.text}");
				onComplete?.Invoke(false);
			}
		}
	}

	// ==========================================================
	// 🔹 DELETE /gesture/{gesture_label}
	// ==========================================================
	public IEnumerator RemoveGesture(string gestureLabel, Action<bool> onComplete = null)
	{
		string url = $"{baseUrl}/gesture/{gestureLabel}";
		using (UnityWebRequest www = UnityWebRequest.Delete(url))
		{
			yield return www.SendWebRequest();

			if (www.result == UnityWebRequest.Result.Success)
			{
				AddGestureResponse response = JsonConvert.DeserializeObject<AddGestureResponse>(www.downloadHandler.text);
				bool success = response.status_code == 0;

				Debug.Log($"RemoveGesture response: {response.message} (Success: {success})");
				onComplete?.Invoke(success);
			}
			else
			{
				Debug.LogError($"Error removing gesture '{gestureLabel}': {www.error}\n{www.downloadHandler.text}");
				onComplete?.Invoke(false);
			}
		}
	}

	// ==========================================================
	// 🔹 DELETE /gesture/ (remove all gestures)
	// ==========================================================
	public IEnumerator RemoveAllGestures(Action<bool> onComplete = null)
	{
		string url = $"{baseUrl}/gesture/";
		using (UnityWebRequest www = UnityWebRequest.Delete(url))
		{
			yield return www.SendWebRequest();

			if (www.result == UnityWebRequest.Result.Success)
			{
				AddGestureResponse response = JsonConvert.DeserializeObject<AddGestureResponse>(www.downloadHandler.text);
				bool success = response.status_code == 0;

				Debug.Log($"RemoveAllGestures response: {response.message} (Success: {success})");
				onComplete?.Invoke(success);
			}
			else
			{
				Debug.LogError($"Error removing all gestures: {www.error}\n{www.downloadHandler.text}");
				onComplete?.Invoke(false);
			}
		}
	}

	// ==========================================================
	// 🔹 GET /get_gestures
	// ==========================================================
	public IEnumerator GetGestures(Action<string> onComplete)
	{
		using (UnityWebRequest www = UnityWebRequest.Get($"{baseUrl}/get_gestures"))
		{
			yield return www.SendWebRequest();

			if (www.result == UnityWebRequest.Result.Success)
			{
				Debug.Log($"Gestures fetched: {www.downloadHandler.text}");
				onComplete?.Invoke(www.downloadHandler.text);
			}
			else
			{
				Debug.LogError($"Error fetching gestures: {www.error}");
				onComplete?.Invoke(null);
			}
		}
	}
}