using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class GesturePracticeManager : MonoBehaviour
{
	private string scriptName = "GesturePracticeManager";

	[SerializeField] private Transform UIContainer;
	[SerializeField] private string UIContainerName = "UI";

	[Header("Navigation Buttons")]
	[SerializeField] private Button backButton;
	[SerializeField] private string backButtonName = "BackButton";

	[Header("Gesture Visualization")]
	[SerializeField] private Dictionary<string, Button> gestureVisualizationButtons;
	[SerializeField] private GameObject gestureVisualizationButtonContainer;
	[SerializeField] private string visButtonContainerName = "GestureVisualizationButtonContainer";
	[SerializeField] private string gesturePrefabPath = "Prefabs/GestureVisualizationUI";
	[SerializeField] private GameObject gestureVisualizationPrefab;
	[SerializeField] private GestureVisualizationUI gestureVisualizationUI;
	[SerializeField] private Button visualizeGestureButton;
	[SerializeField] private string visualizeGestureButtonName = "VisualizeGestureButton";
	[SerializeField] private Button closeVisualizationButton;
	[SerializeField] private string closeVisualizationButtonName = "CloseVisualizationButton";
	[SerializeField] private Button replayVisualizationButton;
	[SerializeField] private string replayVisualizationButtonName = "ReplayVisualizationButton";
	[SerializeField] private GameObject closedVisualizationUIElements;
	[SerializeField] private string closedVisualizationUIElementsName = "ClosedVisualizationUIElements";
	[SerializeField] private GameObject openedVisualizationUIElements;
	[SerializeField] private string openedVisualizationUIElementsName = "OpenedVisualizationUIElements";


	private Color selectedColor = Color.green;
	private Color unselectedColor = Color.white;
	private string selectedGestureKey;

	void Awake()
	{
		GestureSystemManager.instance.StartGestureRecognition();

		if (UIContainer == null)
			UIContainer = GameObject.Find(UIContainerName).transform;
		if (UIContainer == null)
			Debug.LogError($"[{scriptName}] UI container not found!");


		if (backButton == null)
			backButton = GameObject.Find(backButtonName).GetComponent<Button>();
		if (backButton == null)
			Debug.LogError($"[{scriptName}] BackButton not found!");

		if (visualizeGestureButton == null)
			visualizeGestureButton = GameObject.Find(visualizeGestureButtonName).GetComponent<Button>();
		if (visualizeGestureButton == null)
			Debug.LogError($"[{scriptName}] VisualizeGestureButton not found!");

		if (closeVisualizationButton == null)
			closeVisualizationButton = GameObject.Find(closeVisualizationButtonName).GetComponent<Button>();
		if (closeVisualizationButton == null)
			Debug.LogError($"[{scriptName}] CloseVisualizationButton not found!");

		if (replayVisualizationButton == null)
			replayVisualizationButton = GameObject.Find(replayVisualizationButtonName).GetComponent<Button>();
		if (replayVisualizationButton == null)
			Debug.LogError($"[{scriptName}] ReplayVisualizationButton not found!");

		if (closedVisualizationUIElements == null)
			closedVisualizationUIElements = GameObject.Find(closedVisualizationUIElementsName);
		if (closedVisualizationUIElements == null)
			Debug.LogError($"[{scriptName}] ClosedVisualizationUIElements not found!");

		if (openedVisualizationUIElements == null)
			openedVisualizationUIElements = GameObject.Find(openedVisualizationUIElementsName);
		if (openedVisualizationUIElements == null)
			Debug.LogError($"[{scriptName}] OpenedVisualizationUIElements not found!");

		if (gestureVisualizationButtonContainer == null)
			gestureVisualizationButtonContainer = GameObject.Find(visButtonContainerName);
		if (gestureVisualizationButtonContainer == null)
			Debug.LogError($"[{scriptName}] GestureVisualizationButtonContainer not found!");
		else
		{
			StartCoroutine(WaitForGestureVisualizationContainer());
		}

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
		}
	}

	void Start()
	{
		backButton.onClick.AddListener(OnBackButtonPressed);
		visualizeGestureButton.onClick.AddListener(OnVisualizeGesturePressed);


		closeVisualizationButton.onClick.AddListener(OnCloseVisualizationPressed);
		replayVisualizationButton.onClick.AddListener(OnReplayVisualizationPressed);

		OpenGestureVisualizationSelection();
	}

	#region Initialization and Menus

	private IEnumerator WaitForGestureVisualizationContainer()
	{
		// Keep trying until the container exists
		while (GestureSystemManager.instance.StoredGestures == null)
		{
			yield return null;
		}

		// Once found, initialize
		InitializeGestureVisualization();
		SetSelectedGestureKey(GestureSystemManager.instance.StoredGestures.Keys.FirstOrDefault());
	}


	private void InitializeGestureVisualization()
	{
		gestureVisualizationButtons = new Dictionary<string, Button>();

		foreach (Transform child in gestureVisualizationButtonContainer.transform)
		{
			Destroy(child.gameObject);
		}

		foreach (string gestureKey in GestureSystemManager.instance.StoredGestures.Keys.ToList())
		{

			// Create a new button object
			GameObject buttonObj = new GameObject($"{gestureKey} Button", typeof(RectTransform), typeof(Button), typeof(Image));
			buttonObj.transform.SetParent(gestureVisualizationButtonContainer.transform, false);

			// Set height of button
			RectTransform buttonTransform = buttonObj.GetComponent<RectTransform>();
			Vector2 size = buttonTransform.sizeDelta;
			size.y = 40f;  // height
			buttonTransform.sizeDelta = size;


			// Add Text component
			GameObject textObj = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI));
			textObj.transform.SetParent(buttonObj.transform, false);
			TextMeshProUGUI buttonText = textObj.GetComponent<TextMeshProUGUI>();
			buttonText.text = gestureKey;
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
			button.onClick.AddListener(() => OnGestureKeyButtonPressed(gestureKey));
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


			gestureVisualizationButtons[gestureKey] = button;
		}
	}

	private void OpenGestureVisualizationSelection()
	{
		gestureVisualizationUI.gameObject.SetActive(false);
		openedVisualizationUIElements.SetActive(false);
		closedVisualizationUIElements.SetActive(true);
	}

	private void OpenGestureVisualizationSelected()
	{
		closedVisualizationUIElements.SetActive(false);
		openedVisualizationUIElements.SetActive(true);
		gestureVisualizationUI.gameObject.SetActive(true);
	}

	private void OnDestroy()
	{
		GestureSystemManager.instance.EndGestureRecognition();
	}

	#endregion

	#region Button Handlers

	private void OnBackButtonPressed()
	{
		Debug.Log("Returning to Main Menu scene!");
		SceneTransitionManager.instance.LoadScene("MainMenu");
	}

	public void OnVisualizeGesturePressed()
	{
		if (selectedGestureKey == null)
		{
			Debug.LogError("No gesture selected for visualization!");
			return;
		}
		else if (!GestureSystemManager.instance.StoredGestures.ContainsKey(selectedGestureKey))
		{
			Debug.LogError("Selected gesture key did not have stored gesture data!");
			return;
		}

		BaseGestureData selectedGestureData = GestureSystemManager.instance.StoredGestures[selectedGestureKey];

		OpenGestureVisualizationSelected();

		Debug.Log($"Visualizing gesture with label '{selectedGestureKey}'!");

		// Initialize with the selected gesture data
		gestureVisualizationUI.Initialize(selectedGestureData);

		// Start playback automatically
		gestureVisualizationUI.StartVisualizationVideo();
	}

	private void OnCloseVisualizationPressed()
	{
		Debug.Log("Closing visualization!");

		gestureVisualizationUI.ResetVideo();
		OpenGestureVisualizationSelection();
	}

	private void OnReplayVisualizationPressed()
	{
		Debug.Log("Replaying the visualization!");

		gestureVisualizationUI.ResetVideo();
		gestureVisualizationUI.StartVisualizationVideo();
	}

	private void OnGestureKeyButtonPressed(string gestureKey)
	{
		Debug.Log($"[{scriptName}] Using default system now!");
		SetSelectedGestureKey(gestureKey);
	}

	/// <summary>
	/// Updates the selected gesture key and changes visuals.
	/// </summary>
	/// <param name="selectedGestureKey">The gesture key to select for the next visualization.</param>
	private void SetSelectedGestureKey(string selectedGestureKey)
	{
		this.selectedGestureKey = selectedGestureKey;

		foreach (string gestureKey in gestureVisualizationButtons.Keys.ToList())
		{
			Button button = gestureVisualizationButtons[gestureKey];
			button.image.color = gestureKey == selectedGestureKey ? selectedColor : unselectedColor;
		}
	}

	#endregion

}
