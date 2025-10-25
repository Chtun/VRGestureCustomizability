using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
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

	// ==========================================================
	// 🔹 Internal data classes for request and response
	// ==========================================================
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
			List<List<float>> rightWrist)
		{
			this.label = label;
			this.left_joints = leftJoints;
			this.right_joints = rightJoints;
			this.left_wrist = leftWrist;
			this.right_wrist = rightWrist;
		}
	}

	[Serializable]
	public class GestureResponse
	{
		public string status;
		public string message;
		public int total_examples;
	}

	// ==========================================================
	// 🔹 POST /add_gesture
	// ==========================================================
	public IEnumerator AddGesture(GestureInput gesture, Action<string> onComplete = null)
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
				Debug.Log($"Gesture added: {www.downloadHandler.text}");
				onComplete?.Invoke(www.downloadHandler.text);
			}
			else
			{
				Debug.LogError($"Error adding gesture: {www.error}\n{www.downloadHandler.text}");
				onComplete?.Invoke(null);
			}
		}
	}

	// ==========================================================
	// 🔹 GET /gestures
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
