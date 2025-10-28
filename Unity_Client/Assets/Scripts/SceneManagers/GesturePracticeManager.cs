using UnityEngine;
using UnityEngine.UI;

public class GesturePracticeManager : MonoBehaviour
{
	[SerializeField] private GestureSystemManager _gestureSystemManager;
	[SerializeField] private SceneTransitionManager _sceneTransitionManager;

	private string scriptName = "GesturePracticeManager";

	[Header("Navigation Buttons")]
	[SerializeField] private Button backButton;
	[SerializeField] private string backButtonName = "BackButton";

	void Awake()
	{
		if (_gestureSystemManager == null)
			_gestureSystemManager = FindFirstObjectByType<GestureSystemManager>();
		if (_gestureSystemManager == null)
			Debug.LogError($"[{scriptName}] GestureSystemManager not found!");
		else
		{
			_gestureSystemManager.StartGestureRecognition();
		}

		if (_sceneTransitionManager == null)
			_sceneTransitionManager = FindFirstObjectByType<SceneTransitionManager>();
		if (_sceneTransitionManager == null)
			Debug.LogError($"[{scriptName}] SceneTransitionManager not found!");

		if (backButton == null)
			backButton = GameObject.Find(backButtonName).GetComponent<Button>();
		if (backButton == null)
			Debug.LogError($"[{scriptName}] BackButton not found!");
	}

	void Start()
	{
		backButton.onClick.AddListener(OnBackButtonPressed);
	}

	private void OnBackButtonPressed()
	{
		Debug.Log("Returning to Main Menu scene!");
		_sceneTransitionManager.LoadScene("MainMenu");
	}

	private void OnDestroy()
	{
		_gestureSystemManager.EndGestureRecognition();
	}

}
