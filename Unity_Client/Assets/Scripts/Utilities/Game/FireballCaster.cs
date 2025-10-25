using UnityEngine;

public class FireballCaster : MonoBehaviour
{
	public GameObject fireballPrefab;
	public GameObject impactEffectPrefab;
	public Transform spawnPoint;
	public float fireballSpeed = 25f;
	public float lifetime = 5f;
	public float cooldown = 300f;

	private InputManager inputManager; // Reference to the Input Manager
	private float lastFireTime = -Mathf.Infinity;

	void Awake()
	{
		inputManager = FindFirstObjectByType<InputManager>();
		if (inputManager == null)
		{
			Debug.LogError("InputManager not found! FireballCaster cannot subscribe to input events.");
			enabled = false;
		}
	}

	void OnEnable()
	{
		if (inputManager != null)
		{
			inputManager.OnFireballCast += TryCastFireball;
		}
	}

	void OnDisable()
	{
		if (inputManager != null)
		{
			inputManager.OnFireballCast -= TryCastFireball;
		}
	}

	private void TryCastFireball()
	{
		float timeSinceLast = Time.time - lastFireTime;

		if (timeSinceLast >= cooldown)
		{
			Debug.Log($"[Fireball] Casting fireball at {Time.time:F2} (cooldown met)");
			CastFireball();
			lastFireTime = Time.time;
		}
		else
		{
			float remaining = cooldown - timeSinceLast;
			Debug.Log($"[Fireball] Still cooling down ({remaining:F1}s remaining)");
		}
	}

	void CastFireball()
	{
		if (fireballPrefab == null || spawnPoint == null)
		{
			Debug.LogWarning("Missing prefab or spawn point!");
			return;
		}

		GameObject fireball = Instantiate(fireballPrefab, spawnPoint.position, spawnPoint.rotation);

		var collision = fireball.GetComponent<FireballCollision>();
		if (collision != null && impactEffectPrefab != null)
			collision.impactEffectPrefab = impactEffectPrefab;

		var rb = fireball.GetComponent<Rigidbody>();
		if (rb != null)
		{
			Transform cameraTransform = Camera.main.transform;
			Vector3 shootDirection = cameraTransform.forward;
			rb.linearVelocity = shootDirection * fireballSpeed;
		}

		Destroy(fireball, lifetime);
	}
}
