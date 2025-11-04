using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GestureSystemManager : MonoBehaviour
{
	[SerializeField] private GestureHTTPClient gestureHTTPClient;
	[SerializeField] private GestureWebSocketStreamer gestureWebSocketStreamer;
	[SerializeField] private JointDataGather jointDataGather;
	[SerializeField] private GestureRecognitionUI gestureRecognitionUI;
	[SerializeField] private InputManager inputManager;

	[SerializeField] private bool recordJointData;
	[SerializeField] public bool useDefaultSystem;

	private string scriptName = "GestureSystemManager";

	public static GestureSystemManager instance; // Singleton reference

	[SerializeField]
	private Dictionary<string, ActionType> GestureKeyToActionType;

	private Dictionary<string, StoredGesture> _storedGestures;
	public Dictionary<string, StoredGesture> StoredGestures => _storedGestures;


	private string currentRecordedGestureKey;

	void Awake()
	{
		// Singleton pattern to ensure only one instance
		if (instance != null && instance != this)
		{
			Destroy(gameObject); // Destroy duplicates
			return;
		}
		instance = this;
		DontDestroyOnLoad(gameObject); // Make persistent

		InitializeGestureKeyToActionType();

		// Auto-find clients if not assigned
		if (gestureHTTPClient == null)
			gestureHTTPClient = FindFirstObjectByType<GestureHTTPClient>();
		if (gestureHTTPClient == null)
			Debug.LogError($"[{scriptName}] GestureHTTPClient not found!");

		if (gestureWebSocketStreamer == null)
			gestureWebSocketStreamer = FindFirstObjectByType<GestureWebSocketStreamer>();
		if (gestureWebSocketStreamer == null)
			Debug.LogError($"[{scriptName}] GestureWebSocketStreamer not found!");

		if (jointDataGather == null)
			jointDataGather = FindFirstObjectByType<JointDataGather>();
		if (jointDataGather == null)
			Debug.LogError($"[{scriptName}] JointDataGather not found!");

		if (gestureWebSocketStreamer != null)
		{
			gestureWebSocketStreamer.OnGestureDataReceived -= HandleGestureMessage;
			gestureWebSocketStreamer.OnGestureDataReceived += HandleGestureMessage;
		}

		// Subscribe to scene load events
		SceneManager.sceneLoaded += OnSceneLoaded;
		RefreshSceneReferences();

		if (inputManager == null)
			inputManager = FindFirstObjectByType<InputManager>();
		if (inputManager == null)
			Debug.LogError($"[{scriptName}] InputManager not found!");

		// If recording data, start a coroutine that waits for tracking before recording
		if (jointDataGather != null && recordJointData)
			StartCoroutine(jointDataGather.WaitForHandsThenRecord());
	}

	void OnDestroy()
	{
		SceneManager.sceneLoaded -= OnSceneLoaded;
	}

	private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
	{
		// Whenever a new scene is loaded, re-fetch scene-specific references
		RefreshSceneReferences();
	}

	private void RefreshSceneReferences()
	{
		// Only refresh the GestureRecognitionUI here, because it's scene-specific
		if (gestureRecognitionUI == null)
		{
			gestureRecognitionUI = FindFirstObjectByType<GestureRecognitionUI>();
			if (gestureRecognitionUI == null)
				Debug.LogWarning($"[{scriptName}] GestureRecognitionUI not found in the current scene!");
			else
				Debug.Log($"[{scriptName}] GestureRecognitionUI found and assigned for the current scene.");
		}
	}

	private void InitializeGestureKeyToActionType()
	{
		// Add default gestures to the key to action dictionary
		GestureKeyToActionType = new Dictionary<string, ActionType>();
		Config config = Config.LoadConfig();

		foreach (string gestureKey in config.defaultGestures.Keys.ToList())
		{
			ActionType associatedActionType = config.defaultGestures[gestureKey];
			GestureKeyToActionType[gestureKey] = associatedActionType;

			Debug.Log($"Added gesture key: {gestureKey} to dictionary with associated action: {associatedActionType.ToString()}");
		}

		foreach (ActionType actionType in Enum.GetValues(typeof(ActionType)))
		{
			GestureKeyToActionType[InputManager.ActionTypeName(actionType)] = actionType;
		}
	}

	#region API

	public bool RefreshGestureList()
	{

		if (gestureHTTPClient != null)
		{
			StartCoroutine(gestureHTTPClient.GetGestures(useDefaultSystem, OnGesturesFetched));
			return true;
		}


		return false;
	}

	public bool IsRecording()
	{
		return jointDataGather.IsRecording;
	}

	public string GetNextGestureKey(ActionType actionType)
	{
		int index = 1;
		string gestureKey = $"{InputManager.ActionTypeName(actionType)} {index}";

		while (GestureKeyToActionType.ContainsKey(gestureKey))
		{
			index += 1;
			gestureKey = $"{InputManager.ActionTypeName(actionType)} {index}";
		}

		return gestureKey;
	}

	public bool StartGestureRecognition()
	{
		if (RefreshGestureList() && gestureWebSocketStreamer != null)
		{
			gestureWebSocketStreamer.Connect(useDefaultSystem);

			return true;
		}

		return false;
	}

	public bool EndGestureRecognition()
	{
		if (gestureWebSocketStreamer != null)
		{
			gestureWebSocketStreamer.Disconnect();

			return true;
		}

		return false;
	}

	public bool StartRecordingGesture(string label)
	{
		if (jointDataGather == null)
		{
			return false;
		}

		if (jointDataGather.IsRecording)
		{
			Debug.LogWarning($"[{scriptName}] Joint data was already recording data, stopping to now record current gesture '{label}'.");
			jointDataGather.StopRecording();
		}

		currentRecordedGestureKey = label;

		Debug.Log($"[{scriptName}] Beginning new recording for gesture '{label}'.");

		StartCoroutine(jointDataGather.WaitForHandsThenRecord(label));

		return true;
	}

	public bool StopRecordingGesture(out GestureInput gestureInput)
	{
		if (string.IsNullOrEmpty(currentRecordedGestureKey) || jointDataGather == null || !jointDataGather.IsRecording)
		{
			gestureInput = null;
			return false;
		}

		jointDataGather.StopRecording();

		string filePath = JointDataGather.GetRecordedCSVPath(currentRecordedGestureKey);
		gestureInput = JointDataGather.ReadCSVToGestureInput(filePath, currentRecordedGestureKey);

		currentRecordedGestureKey = string.Empty;

		return true;
	}

	public IEnumerator AddGesture(GestureInput gestureInput, ActionType associatedActionType, Action<bool> onComplete)
	{
		// Validate that the label matches the action type, if already in the mapping.
		if (GestureKeyToActionType.ContainsKey(gestureInput.label) && GestureKeyToActionType[gestureInput.label] != associatedActionType)
		{
			Debug.LogWarning($"Gesture '{gestureInput.label}' is already mapped to a different action type.");
			onComplete?.Invoke(false); // notify caller
			yield break;
		}

		bool success = false;

		// Start the HTTP request and wait for it
		yield return StartCoroutine(gestureHTTPClient.AddGesture(gestureInput, (requestSuccess) =>
		{
			success = requestSuccess;
		}));

		if (success)
		{
			GestureKeyToActionType[gestureInput.label] = associatedActionType;
		}

		// Notify caller when done
		onComplete?.Invoke(success);
	}

	public IEnumerator RemoveGesture(string gestureLabel, Action<bool> onComplete = null)
	{
		bool success = false;

		// Start the HTTP request and wait for it
		yield return StartCoroutine(gestureHTTPClient.RemoveGesture(gestureLabel, (requestSuccess) =>
		{
			success = requestSuccess;
		}));

		if (success)
		{
			GestureKeyToActionType.Remove(gestureLabel);
		}

		// Notify caller when done
		onComplete?.Invoke(success);
	}

	public IEnumerator RemoveAllGestures(Action<bool> onComplete = null)
	{
		bool success = false;

		// Start the HTTP request and wait for it
		yield return StartCoroutine(gestureHTTPClient.RemoveAllGestures((requestSuccess) =>
		{
			success = requestSuccess;
		}));

		if (success)
		{
			InitializeGestureKeyToActionType();
		}
	}

	public bool IsDataReliable(bool isRightHand)
	{
		return jointDataGather.IsDataReliable(isRightHand);
	}

	#endregion API

	#region Handle Incoming Messages

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

		var jsonObj = JObject.Parse(jsonResponse);
		_storedGestures = new Dictionary<string, StoredGesture>();

		foreach (var gestureProperty in jsonObj.Properties())
		{
			string gestureKey = gestureProperty.Name;
			JArray templates = (JArray)gestureProperty.Value;

			// You can loop over all templates if needed; here we just take the first for simplicity.
			var template = templates.First as JObject;

			var storedGesture = new StoredGesture(
				label: gestureKey,
				leftHandVectors: ListUtilities.Parse3DList((JArray)template["left_hand_vectors"]),
				rightHandVectors: ListUtilities.Parse3DList((JArray)template["right_hand_vectors"]),
				leftJointRotations: ListUtilities.Parse3DList((JArray)template["left_joint_rotations"]),
				rightJointRotations: ListUtilities.Parse3DList((JArray)template["right_joint_rotations"]),
				leftWristPositions: ListUtilities.Parse2DList((JArray)template["left_wrist_positions"]),
				rightWristPositions: ListUtilities.Parse2DList((JArray)template["right_wrist_positions"]),
				leftWristRotations: ListUtilities.Parse2DList((JArray)template["left_wrist_rotations"]),
				rightWristRotations: ListUtilities.Parse2DList((JArray)template["right_wrist_rotations"])
			);

			_storedGestures[gestureKey] = storedGesture;
			Debug.Log($"[{scriptName}] Stored gesture '{gestureKey}' with {storedGesture.left_hand_vectors.Count} frames.");
		}

		Debug.Log($"[{scriptName}] Finished loading {_storedGestures.Count} gestures.");
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
			if (gestureRecognitionUI != null)
			{
				if (!gestureRecognitionUI.IsUIInitialized)
				{
					gestureRecognitionUI.InitializeGestures(gestureDict);
				}
				else
				{
					gestureRecognitionUI.UpdateGestures(gestureDict);
				}
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

	#endregion Handle Incoming Messages
}
