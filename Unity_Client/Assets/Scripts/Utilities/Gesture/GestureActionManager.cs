using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

public class GestureActionManager : MonoBehaviour
{
	[SerializeField] private GestureHTTPClient gestureHTTPClient;
	[SerializeField] private GestureWebSocketStreamer gestureWebSocketStreamer;
	[SerializeField] private GestureRecognitionUI gestureRecognitionUI;
	[SerializeField] private InputManager inputManager;

	private bool isUIInitialized = false;
	private string scriptName = "GestureActionManager";

	Dictionary<string, ActionType> GestureKeyToActionType;

	void Awake()
	{
		// Auto-find clients if not assigned
		if (gestureHTTPClient == null)
			gestureHTTPClient = FindFirstObjectByType<GestureHTTPClient>();
		if (gestureHTTPClient == null)
			Debug.LogError($"[{scriptName}] GestureHTTPClient not found!");

		if (gestureWebSocketStreamer == null)
			gestureWebSocketStreamer = FindFirstObjectByType<GestureWebSocketStreamer>();
		if (gestureWebSocketStreamer == null)
			Debug.LogError($"[{scriptName}] GestureWebSocketStreamer not found!");

		if (gestureRecognitionUI == null)
			gestureRecognitionUI = FindFirstObjectByType<GestureRecognitionUI>();
		if (gestureRecognitionUI == null)
			Debug.LogError($"[{scriptName}] GestureRecognitionUI not found!");

		if (gestureWebSocketStreamer != null)
			gestureWebSocketStreamer.OnGestureDataReceived += HandleGestureMessage;

		if (inputManager == null)
			inputManager = FindFirstObjectByType<InputManager>();
		if (inputManager == null)
			Debug.LogError($"[{scriptName}] InputManager not found!");

		GestureKeyToActionType = new Dictionary<string, ActionType>();
	}

	void Start()
	{
		if (gestureHTTPClient != null)
			StartCoroutine(gestureHTTPClient.GetGestures(OnGesturesFetched));
	}

	/// <summary>
	/// Handles when all gestures stored on Gesture Recognition server is fetched.
	/// </summary>
	/// <param name="jsonResponse">The json payload.</param>
	private void OnGesturesFetched(string jsonResponse)
	{
		if (string.IsNullOrEmpty(jsonResponse))
		{
			Debug.LogError($"[{scriptName}] Failed to load gestures.");
			return;
		}

		Debug.Log($"[{scriptName}] Gestures loaded.");

		// Parse JSON
		var jsonObj = JObject.Parse(jsonResponse);

		// Extract all gesture keys
		foreach (var gestureKey in jsonObj.Properties())
		{
			Debug.Log($"Gesture Key: {gestureKey.Name}");


			// Hard code each gesture key to fireball as of now.
			GestureKeyToActionType[gestureKey.Name] = ActionType.CastFireball;
		}

	}

	/// <summary>
	/// Handles an incoming gesture message from Gesture Recognition server.
	/// </summary>
	/// <param name="json">The json payload.</param>
	private void HandleGestureMessage(string json)
	{
		try
		{
			var obj = JObject.Parse(json);

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

			HandleGestureActions(gestureDict);
		}
		catch (Exception ex)
		{
			Debug.LogError($"Failed to parse gesture JSON: {ex}");
		}
	}

	/// <summary>
	/// Handles the incoming gesture matches. If there are any matches, cast the particular action.
	/// </summary>
	/// <param name="gestureDict">The dictionary that contains each gesture key and its distance and whether it is recognized or not.</param>
	private void HandleGestureActions(Dictionary<string, (float distance, bool matched)> gestureDict)
	{
		foreach (string key in gestureDict.Keys)
		{
			if (gestureDict[key].matched == true && GestureKeyToActionType.ContainsKey(key))
			{
				ActionType actionType = GestureKeyToActionType[key];

				inputManager.TakeAction(actionType);
			}
		}
	}
}
