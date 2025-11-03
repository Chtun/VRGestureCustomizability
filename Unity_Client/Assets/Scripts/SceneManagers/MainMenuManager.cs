using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class MainUIController : MonoBehaviour
{
	[Header("Navigation Buttons")]
	[SerializeField] private Button gestureRecordingButton;
	[SerializeField] private Button gesturePracticeButton;
	[SerializeField] private Button gameButton;
	[SerializeField] private Button puzzleButton;

	[Header("Toggle Gesture System Buttons")]
	[SerializeField] private Button defaultSystemButton;
	[SerializeField] private Button userDefinedSystemButton;

	private Color selectedColor = Color.green;
	private Color unselectedColor = Color.white;

	private string scriptName = "MainMenuManager";

	private void Awake()
	{
		// Validate references
		if (gestureRecordingButton == null) Debug.LogWarning("GestureRecording not assigned!");
		if (gesturePracticeButton == null) Debug.LogWarning("GesturePractice not assigned!");
		if (gameButton == null) Debug.LogWarning("GameButton not assigned!");
		if (puzzleButton == null) Debug.LogWarning("PuzzleButton not assigned!");
		if (defaultSystemButton == null) Debug.LogWarning("DefaultToggle not assigned!");
		if (userDefinedSystemButton == null) Debug.LogWarning("UserDefinedToggle not assigned!");
	}
	private void Start()
	{
		// Bind button click events
		gestureRecordingButton.onClick.AddListener(OnGestureRecordingButtonPressed);
		gesturePracticeButton.onClick.AddListener(OnGesturePracticeButtonPressed);
		gameButton.onClick.AddListener(OnGameButtonPressed);
		puzzleButton.onClick.AddListener(OnPuzzleButtonPressed);
		defaultSystemButton.onClick.AddListener(OnDefaultSystemButtonPressed);
		userDefinedSystemButton.onClick.AddListener(OnUserDefinedSystemButtonPressed);

		SetGestureSystemType(GestureSystemManager.instance.useDefaultSystem);
	}

	#region Button Handlers

	private void OnGestureRecordingButtonPressed()
	{
		Debug.Log("Changing to Gesture Recording scene!");

		SceneTransitionManager.instance.LoadScene("GestureRecording");
	}

	private void OnGesturePracticeButtonPressed()
	{
		Debug.Log("Changing to Gesture Practice scene!");
		SceneTransitionManager.instance.LoadScene("GesturePractice");
	}

	private void OnGameButtonPressed()
	{
		Debug.Log("Changing to Game scene!");
		SceneTransitionManager.instance.LoadScene("WizardGame");
	}

	private void OnPuzzleButtonPressed()
	{
		Debug.Log("Changing to Puzzle scene!");
		SceneTransitionManager.instance.LoadScene("Puzzle");
	}

	private void OnDefaultSystemButtonPressed()
	{
		Debug.Log("Using default system now!");
		SetGestureSystemType(true);
	}

	private void OnUserDefinedSystemButtonPressed()
	{
		Debug.Log("Using user-defined system now!");
		SetGestureSystemType(false);
	}

	#endregion

	/// <summary>
	/// Updates the useDefaultSystem flag and changes button visuals.
	/// </summary>
	/// <param name="useDefault">If true, selects default system; otherwise user-defined.</param>
	private void SetGestureSystemType(bool useDefault)
	{
		if (GestureSystemManager.instance != null)
		{
			GestureSystemManager.instance.useDefaultSystem = useDefault;

			// Deselect the button immediately so it doesn't stay bright
			EventSystem.current.SetSelectedGameObject(null);

			// Update button colors to indicate selection
			defaultSystemButton.image.color = GestureSystemManager.instance.useDefaultSystem ? selectedColor : unselectedColor;
			userDefinedSystemButton.image.color = GestureSystemManager.instance.useDefaultSystem ? unselectedColor : selectedColor;
		}
		else
		{
			Debug.LogError($"[{scriptName}] Gesture system type could not be set since GestureSystemManager reference is null!");
		}
	}
}