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
		_gestureSystemManager.StartGestureRecognition();
	}

	private void OnDestroy()
	{
		_gestureSystemManager.EndGestureRecognition();
	}
}
