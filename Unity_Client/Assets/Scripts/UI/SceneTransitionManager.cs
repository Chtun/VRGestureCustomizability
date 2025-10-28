using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneTransitionManager : MonoBehaviour
{
	public static SceneTransitionManager Instance { get; private set; }

	[Header("OVR Screen Fade Settings")]
	[SerializeField] private float defaultFadeDuration = 0.5f;

	private void Awake()
	{
		// Singleton pattern
		if (Instance != null && Instance != this)
		{
			Destroy(gameObject);
			return;
		}
		Instance = this;
		DontDestroyOnLoad(gameObject);
	}

	/// <summary>
	/// Call this to load a new scene with fade in/out.
	/// </summary>
	/// <param name="sceneName">Scene to load</param>
	/// <param name="fadeDuration">Optional fade duration</param>
	public void LoadScene(string sceneName, float fadeDuration = -1f)
	{
		if (fadeDuration < 0f) fadeDuration = defaultFadeDuration;
		StartCoroutine(FadeAndLoad(sceneName, fadeDuration));
	}

	private IEnumerator FadeAndLoad(string sceneName, float fadeDuration)
	{
		OVRScreenFade.instance.fadeTime = fadeDuration;
		// Fade out
		OVRScreenFade.instance.FadeOut();
		yield return new WaitForSeconds(fadeDuration);

		// Load scene
		SceneManager.LoadScene(sceneName);
		yield return null; // wait one frame to ensure scene is loaded

		// Fade in
		OVRScreenFade.instance.FadeIn();
	}
}
