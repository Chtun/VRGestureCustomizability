using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class MainUIController : MonoBehaviour
{
	[Header("Gesture Scene Buttons")]
	[SerializeField] private Button gestureRecordingButton;
	[SerializeField] private Button gesturePracticeButton;

	[Header("Game & Puzzle Scene Buttons")]
	[SerializeField] private Button gameButton;
	[SerializeField] private Button puzzleButton;

	[Header("Toggle System Buttons")]
	[SerializeField] private Button defaultSystemButton;
	[SerializeField] private Button userDefinedSystemButton;

	private Color selectedColor = Color.green;
	private Color unselectedColor = Color.white;

	private bool useDefaultSystem;

	private void Awake()
	{
		// Optional: Validate references
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

		SetUseDefaultSystem(true);
	}

	#region Button Handlers
	private void OnGestureRecordingButtonPressed()
	{
		Debug.Log("Changing to Gesture Recording scene!");
		// TODO: Change scene
	}

	private void OnGesturePracticeButtonPressed()
	{
		Debug.Log("Changing to Gesture Practice scene!");
		// TODO: Change scene
	}

	private void OnGameButtonPressed()
	{
		Debug.Log("Changing to Game scene!");
		// TODO: Change scene
	}

	private void OnPuzzleButtonPressed()
	{
		Debug.Log("Changing to Puzzle scene!");
		// TODO: Change scene
	}

	private void OnDefaultSystemButtonPressed()
	{
		Debug.Log("Using default system now!");
		SetUseDefaultSystem(true);
	}

	private void OnUserDefinedSystemButtonPressed()
	{
		Debug.Log("Using user-defined system now!");
		SetUseDefaultSystem(false);
	}

	#endregion

	/// <summary>
	/// Updates the useDefaultSystem flag and changes button visuals.
	/// </summary>
	/// <param name="useDefault">If true, selects default system; otherwise user-defined.</param>
	private void SetUseDefaultSystem(bool useDefault)
	{
		useDefaultSystem = useDefault;

		// Deselect the button immediately so it doesn't stay bright
		EventSystem.current.SetSelectedGameObject(null);

		// Update button colors to indicate selection
		defaultSystemButton.image.color = useDefaultSystem ? selectedColor : unselectedColor;
		userDefinedSystemButton.image.color = useDefaultSystem ? unselectedColor : selectedColor;
	}
}