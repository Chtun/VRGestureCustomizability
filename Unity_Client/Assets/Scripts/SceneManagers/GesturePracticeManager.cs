using UnityEngine;
using UnityEngine.UI;

public class GesturePracticeManager : MonoBehaviour
{
	private string scriptName = "GesturePracticeManager";

	[Header("Navigation Buttons")]
	[SerializeField] private Button backButton;
	[SerializeField] private string backButtonName = "BackButton";

	void Awake()
	{
		GestureSystemManager.instance.StartGestureRecognition();

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
		SceneTransitionManager.instance.LoadScene("MainMenu");
	}

	private void OnDestroy()
	{
		GestureSystemManager.instance.EndGestureRecognition();
	}

}
