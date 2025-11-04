using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.UI;

public class GestureRecordingManager : MonoBehaviour
{
	private string scriptName = "GestureRecordingManager";

	[SerializeField] private Transform UIContainer;
	[SerializeField] private string UIContainerName = "UI";

	[Header("Navigation Buttons")]
	[SerializeField] private Button backButton;
	[SerializeField] private string backButtonName = "BackButton";

	[Header("Recording Menu Elements")]
	[SerializeField] private Canvas recordingMenuCanvas;
	[SerializeField] private string recordingMenuCanvasName = "RecordingMenuCanvas";
	[SerializeField] private Button addGestureButton;
	[SerializeField] private string addGestureButtonName = "AddGestureButton";
	[SerializeField] private Button removeGesturesButton;
	[SerializeField] private string removeGesturesButtonName = "RemoveCustomGesturesButton";
	[SerializeField] private GameObject actionTypeSelectionContainer;
	[SerializeField] private string selectionContainerName = "ActionTypeSelectionContainer";
	[SerializeField] private Dictionary<ActionType, Button> actionTypeSelectionButtons;

	[Header("Data Display Elements")]
	[SerializeField] private Canvas dataDisplayCanvas;
	[SerializeField] private string dataDisplayCanvasName = "DataDisplayCanvas";
	[SerializeField] private TextMeshProUGUI isRecordingText;
	[SerializeField] private string isRecordingTextName = "IsRecordingText";
	[SerializeField] private TextMeshProUGUI recordingNameText;
	[SerializeField] private string recordingNameTextName = "RecordingNameText";
	[SerializeField] private Button confirmRecordingButton;
	[SerializeField] private string confirmRecordingButtonName = "ConfirmRecordingButton";
	[SerializeField] private Button redoRecordingButton;
	[SerializeField] private string redoRecordingButtonName = "RedoRecordingButton";
	[SerializeField] private string gesturePrefabPath = "Prefabs/GestureVisualizationUI";
	[SerializeField] private GameObject gestureVisualizationPrefab;
	[SerializeField] private GestureVisualizationUI gestureVisualizationUI;

	private GestureInput _currentRecordedGesture;
	public GestureInput CurrentRecordedGesture
	{
		get => _currentRecordedGesture;
		set
		{
			_currentRecordedGesture = value;

			// Enable the confirm button only if there is a recorded gesture
			if (confirmRecordingButton != null)
			{
				confirmRecordingButton.interactable = _currentRecordedGesture != null;
				gestureVisualizationUI.StopVisualizationVideo();
				gestureVisualizationUI.gameObject.SetActive(false);
			}
		}
	}


	private Color selectedColor = Color.green;
	private Color unselectedColor = Color.white;
	private ActionType selectedActionType;


	void Awake()
	{
		if (backButton == null)
			backButton = GameObject.Find(backButtonName).GetComponent<Button>();
		if (backButton == null)
			Debug.LogError($"[{scriptName}] BackButton not found!");

		if (recordingMenuCanvas == null)
			recordingMenuCanvas = GameObject.Find(recordingMenuCanvasName).GetComponent<Canvas>();
		if (recordingMenuCanvas == null)
			Debug.LogError($"[{scriptName}] RecordingMenuCanvas not found!");

		if (addGestureButton == null)
			addGestureButton = GameObject.Find(addGestureButtonName).GetComponent<Button>();
		if (addGestureButton == null)
			Debug.LogError($"[{scriptName}] AddGestureButton not found!");

		if (removeGesturesButton == null)
			removeGesturesButton = GameObject.Find(removeGesturesButtonName).GetComponent<Button>();
		if (removeGesturesButton == null)
			Debug.LogError($"[{scriptName}] RemoveCustomGesturesButton not found!");

		if (actionTypeSelectionContainer == null)
			actionTypeSelectionContainer = GameObject.Find(selectionContainerName);
		if (actionTypeSelectionContainer == null)
			Debug.LogError($"[{scriptName}] ActionTypeSelectionContainer not found!");
		else
		{
			InitializeActionTypeSelection();
		}

		if (dataDisplayCanvas == null)
			dataDisplayCanvas = GameObject.Find(dataDisplayCanvasName).GetComponent<Canvas>();
		if (dataDisplayCanvas == null)
			Debug.LogError($"[{scriptName}] DataDisplayCanvas not found!");

		if (isRecordingText == null)
			isRecordingText = GameObject.Find(isRecordingTextName).GetComponent<TextMeshProUGUI>();
		if (isRecordingText == null)
			Debug.LogError($"[{scriptName}] IsRecordingText not found!");
		else
		{
			isRecordingText.text = "Is Recording:";
		}

		if (recordingNameText == null)
			recordingNameText = GameObject.Find(recordingNameTextName).GetComponent<TextMeshProUGUI>();
		if (recordingNameText == null)
			Debug.LogError($"[{scriptName}] RecordingNameText not found!");
		else
		{
			recordingNameText.text = $"Recording:\n{InputManager.ActionTypeName(selectedActionType)}";
		}

		if (confirmRecordingButton == null)
			confirmRecordingButton = GameObject.Find(confirmRecordingButtonName).GetComponent<Button>();
		if (confirmRecordingButton == null)
			Debug.LogError($"[{scriptName}] ConfirmRecordingButton not found!");

		if (redoRecordingButton == null)
			redoRecordingButton = GameObject.Find(redoRecordingButtonName).GetComponent<Button>();
		if (redoRecordingButton == null)
			Debug.LogError($"[{scriptName}] RedoRecordingButton not found!");

		// Load prefab dynamically
		if (gestureVisualizationPrefab == null)
		{
			gestureVisualizationPrefab = Resources.Load<GameObject>(gesturePrefabPath);

			if (gestureVisualizationPrefab == null)
			{
				Debug.LogError($"Failed to load Gesture Visualization Prefab from path '{gesturePrefabPath}'");
				return;
			}

			// Instantiate the prefab
			GameObject visualizationGO = Instantiate(
				gestureVisualizationPrefab,
				UIContainer != null ? UIContainer : null
			);

			// Get the GestureVisualizationUI component
			gestureVisualizationUI = visualizationGO.GetComponent<GestureVisualizationUI>();
			if (gestureVisualizationUI == null)
			{
				Debug.LogError("Prefab is missing GestureVisualizationUI component!");
				return;
			}

			gestureVisualizationUI.gameObject.SetActive(false);
		}

		OpenRecordingMenu();
		SetSelectedActionType(ActionType.CastFireball);
	}

	void Start()
	{
		backButton.onClick.AddListener(OnBackButtonPressed);
		addGestureButton.onClick.AddListener(OnAddGestureButtonPressed);
		confirmRecordingButton.onClick.AddListener(OnConfirmButtonPressed);
		redoRecordingButton.onClick.AddListener(OnRedoButtonPressed);
		removeGesturesButton.onClick.AddListener(OnRemoveAllGesturesButtonPressed);
	}

	void Update()
	{
		if (Keyboard.current.spaceKey.wasPressedThisFrame)
		{
			OnSpaceBarPressed();
		}
	}

	#region Initialization and Menus

	private void InitializeActionTypeSelection()
	{
		actionTypeSelectionButtons = new Dictionary<ActionType, Button>();

		foreach (Transform child in actionTypeSelectionContainer.transform)
		{
			Destroy(child.gameObject);
		}

		foreach (ActionType actionType in Enum.GetValues(typeof(ActionType)))
		{
			if (actionType.Equals(ActionType.Inactivated))
				continue;

			// Create a new button object
			GameObject buttonObj = new GameObject(actionType.ToString(), typeof(RectTransform), typeof(Button), typeof(Image));
			buttonObj.transform.SetParent(actionTypeSelectionContainer.transform, false);

			// Set height of button
			RectTransform buttonTransform = buttonObj.GetComponent<RectTransform>();
			Vector2 size = buttonTransform.sizeDelta;
			size.y = 40f;  // height
			buttonTransform.sizeDelta = size;


			// Add Text component
			GameObject textObj = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI));
			textObj.transform.SetParent(buttonObj.transform, false);
			TextMeshProUGUI buttonText = textObj.GetComponent<TextMeshProUGUI>();
			buttonText.text = InputManager.ActionTypeName(actionType);
			buttonText.alignment = TextAlignmentOptions.Midline;
			buttonText.color = Color.white;
			buttonText.fontSize = 20;

			RectTransform textRect = textObj.GetComponent<RectTransform>();
			textRect.anchorMin = Vector2.zero;    // Bottom-left corner
			textRect.anchorMax = Vector2.one;     // Top-right corner
			textRect.offsetMin = Vector2.zero;    // No offset
			textRect.offsetMax = Vector2.zero;    // No offset
			textRect.pivot = new Vector2(0.5f, 0.5f);

			// Add a listener to the button.
			Button button = buttonObj.GetComponent<Button>();
			button.onClick.AddListener(() => OnActionTypeButtonPressed(actionType));
			button.image = buttonObj.GetComponent<Image>();

			ColorBlock colors = button.colors;

			// Modify the colors
			Color normalColor;
			ColorUtility.TryParseHtmlString("#555555", out normalColor);
			colors.normalColor = normalColor;      // Normal

			Color highlightedColor;
			ColorUtility.TryParseHtmlString("#909090", out highlightedColor);
			colors.highlightedColor = normalColor;      // When hovered

			Color pressedColor;
			ColorUtility.TryParseHtmlString("#444444", out pressedColor);
			colors.pressedColor = normalColor;      // When clicked

			Color selectedColor;
			ColorUtility.TryParseHtmlString("#F5F5F5", out selectedColor);
			colors.selectedColor = normalColor;      // When selected

			Color disabledColor;
			ColorUtility.TryParseHtmlString("#C8C8C8", out disabledColor);
			disabledColor.a = 128;
			colors.disabledColor = normalColor;      // When disabled

			// Apply it back to the button
			button.colors = colors;


			actionTypeSelectionButtons[actionType] = button;
		}
	}

	private void OpenRecordingMenu()
	{
		dataDisplayCanvas.gameObject.SetActive(false);
		recordingMenuCanvas.gameObject.SetActive(true);
	}

	private void OpenRecordingMode()
	{
		recordingMenuCanvas.gameObject.SetActive(false);
		dataDisplayCanvas.gameObject.SetActive(true);

		CurrentRecordedGesture = null;
	}

	#endregion

	#region Button Handlers

	private void OnBackButtonPressed()
	{
		if (GestureSystemManager.instance.IsRecording())
		{
			GestureInput temp;
			GestureSystemManager.instance.StopRecordingGesture(out temp);
		}

		Debug.Log($"[{scriptName}] Returning to Main Menu scene!");
		SceneTransitionManager.instance.LoadScene("MainMenu");
	}

	private void OnAddGestureButtonPressed()
	{
		Debug.Log($"[{scriptName}] Opening menu for recording new gesture for action type: {selectedActionType}!");

		// Copy transform properties
		dataDisplayCanvas.transform.position = recordingMenuCanvas.transform.position;
		dataDisplayCanvas.transform.rotation = recordingMenuCanvas.transform.rotation;
		dataDisplayCanvas.transform.localScale = recordingMenuCanvas.transform.localScale;

		// Toggle visibility
		OpenRecordingMode();
	}

	private void OnActionTypeButtonPressed(ActionType actionType)
	{
		Debug.Log($"[{scriptName}] Using default system now!");
		SetSelectedActionType(actionType);
	}

	private void OnRemoveAllGesturesButtonPressed()
	{
		Debug.Log($"[{scriptName}] Removing all custom gestures from system.");

		RemoveAllCustomGestures();

		// Deselect the button immediately so it doesn't stay bright
		EventSystem.current.SetSelectedGameObject(null);
	}

	private void OnSpaceBarPressed()
	{
		if (CurrentRecordedGesture != null)
			return;

		if (GestureSystemManager.instance.IsRecording())
		{
			StopRecording();
		}
		else
		{
			StartCoroutine(StartRecordingWithCountdown());
		}
	}

	private IEnumerator StartRecordingWithCountdown()
	{
		// Make sure recording mode (data display) is active
		if (dataDisplayCanvas == null || !dataDisplayCanvas.gameObject.activeSelf)
		{
			Debug.LogWarning($"[{scriptName}] Cannot start recording — recording mode is not active (DataDisplayCanvas is not visible).");
		}

		int countdownTime = 3; // number of seconds before recording starts
		for (int i = countdownTime; i > 0; i--)
		{
			if (isRecordingText != null)
			{
				isRecordingText.text = $"Recording in {i.ToString()}:";
			}
			yield return new WaitForSeconds(1f);
		}

		// Clear the text after countdown
		if (isRecordingText != null)
			isRecordingText.text = "Is Recording:";

		// Now actually start recording
		StartRecording();
	}


	private void OnConfirmButtonPressed()
	{
		ConfirmRecording();
		OpenRecordingMenu();
	}

	private void OnRedoButtonPressed()
	{
		CurrentRecordedGesture = null;

		EventSystem.current.SetSelectedGameObject(null);
	}

	#endregion

	#region Recording Methods

	private void StartRecording()
	{
		// Make sure recording mode (data display) is active
		if (dataDisplayCanvas == null || !dataDisplayCanvas.gameObject.activeSelf)
		{
			Debug.LogWarning($"[{scriptName}] Cannot start recording — recording mode is not active (DataDisplayCanvas is not visible).");
			return;
		}

		string gestureKey = GestureSystemManager.instance.GetNextGestureKey(selectedActionType);
		GestureSystemManager.instance.StartRecordingGesture(gestureKey);
	}

	private void StopRecording()
	{
		GestureInput recordedGesture;

		bool successfulRecording = GestureSystemManager.instance.StopRecordingGesture(out recordedGesture);

		if (successfulRecording)
		{
			recordedGesture.label = InputManager.ActionTypeName(selectedActionType);
			CurrentRecordedGesture = recordedGesture;
			Debug.Log($"[{scriptName}] The recording was stopped successfully.");

			GestureInput copiedRecordedGesture = recordedGesture.DeepCopy();

			copiedRecordedGesture.left_joint_rotations = HandUtilities.ComputeParentRelativeRotations(copiedRecordedGesture.left_joint_rotations);
			copiedRecordedGesture.right_joint_rotations = HandUtilities.ComputeParentRelativeRotations(copiedRecordedGesture.right_joint_rotations);

			gestureVisualizationUI.gameObject.SetActive(true);

			gestureVisualizationUI.Initialize(copiedRecordedGesture);
			gestureVisualizationUI.StartVisualizationVideo(continualLoop: true);
		}
		else
		{
			Debug.LogWarning($"[{scriptName}] Stopping the recording was not successful!");
		}
	}

	private void ConfirmRecording()
	{
		if (CurrentRecordedGesture != null)
		{
			StartCoroutine(GestureSystemManager.instance.AddGesture(
				CurrentRecordedGesture,
				selectedActionType,
				success =>
				{
					if (success)
						Debug.Log($"[{scriptName}] Successfully added gesture for {selectedActionType}!");
					else
						Debug.LogError($"[{scriptName}] Failed to add gesture for {selectedActionType}!");
				}
			));
		}
		else
		{
			Debug.LogError($"[{scriptName}] There is no gesture recording to confirm!");
		}

		CurrentRecordedGesture = null;
		dataDisplayCanvas.gameObject.SetActive(false);
	}

	private void RemoveAllCustomGestures()
	{
		GestureInput recordedGesture;

		bool successfulRecording = GestureSystemManager.instance.StopRecordingGesture(out recordedGesture);

		if (successfulRecording)
		{
			CurrentRecordedGesture = recordedGesture;
		}
		else
		{
			Debug.LogWarning($"[{scriptName}] Stopping the recording was not successful!");
		}
	}

	/// <summary>
	/// Updates the selected action type and changes visuals.
	/// </summary>
	/// <param name="inputActionType">The action type to select for the next recording.</param>
	private void SetSelectedActionType(ActionType inputActionType)
	{
		selectedActionType = inputActionType;

		foreach (ActionType actionType in actionTypeSelectionButtons.Keys.ToList())
		{
			Button button = actionTypeSelectionButtons[actionType];
			button.image.color = actionType == selectedActionType ? selectedColor : unselectedColor;
		}
	}

	#endregion

}
