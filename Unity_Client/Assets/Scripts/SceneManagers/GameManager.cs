using UnityEngine;

public class GameManager : MonoBehaviour
{
	[SerializeField] private GestureSystemManager _gestureSystemManager;

	private string scriptName = "GameManager";

	void Awake()
	{
		// Auto-find gesture system manager
		if (_gestureSystemManager == null)
			_gestureSystemManager = FindFirstObjectByType<GestureSystemManager>();
		if (_gestureSystemManager == null)
			Debug.LogError($"[{scriptName}] GestureSystemManager not found!");
	}

	void Start()
	{
        Config config = Config.LoadConfig();
        string gameTaskRecordingName = config.GetTaskRecordingName();
        Debug.Log($"Recording game task to: {gameTaskRecordingName}");

        _gestureSystemManager.StartGestureRecognition();
		_gestureSystemManager.StartRecordingGesture(gameTaskRecordingName);
		_gestureSystemManager.SetActionLogging(true);
	}

	private void OnDestroy()
	{
		_gestureSystemManager.EndGestureRecognition();
	}
}
