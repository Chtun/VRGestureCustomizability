using UnityEngine;

public class LightningCaster : MonoBehaviour
{
    public GameObject projectilePrefab;
    public GameObject impactEffectPrefab;
    public Transform spawnPoint;
    public float projectileSpeed = 50f;
    public float lifetime = 5f;
    public float cooldown = 2f;

    private InputManager inputManager; // Reference to the Input Manager
    private float lastFireTime = -Mathf.Infinity;

    void Awake()
    {
        inputManager = FindFirstObjectByType<InputManager>();
        if (inputManager == null)
        {
            Debug.LogError("InputManager not found! LightningCaster cannot subscribe to input events.");
            enabled = false;
        }
    }

    void OnEnable()
    {
        if (inputManager != null)
        {
            inputManager.OnLightningCast += TryCastLightning;
        }
    }

    void OnDisable()
    {
        if (inputManager != null)
        {
            inputManager.OnLightningCast -= TryCastLightning;
        }
    }

    private void TryCastLightning()
    {
        float timeSinceLast = Time.time - lastFireTime;

        if (timeSinceLast >= cooldown)
        {
            Debug.Log($"[Lightning] Casting lightning at {Time.time:F2} (cooldown met)");
            CastLightning();
            lastFireTime = Time.time;
        }
        else
        {
            float remaining = cooldown - timeSinceLast;
            Debug.Log($"[Lightning] Still cooling down ({remaining:F1}s remaining)");
        }
    }

    void CastLightning()
    {
        if (projectilePrefab == null || spawnPoint == null)
        {
            Debug.LogWarning("Missing prefab or spawn point!");
            return;
        }

        GameObject lightning = Instantiate(projectilePrefab, spawnPoint.position, spawnPoint.rotation);

        var collision = lightning.GetComponent<LightningCollision>();
        if (collision != null && impactEffectPrefab != null)
            collision.impactEffectPrefab = impactEffectPrefab;

        var rb = lightning.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Transform cameraTransform = Camera.main.transform;
            Vector3 shootDirection = cameraTransform.forward;
            rb.linearVelocity = shootDirection * projectileSpeed;
        }

        Destroy(lightning, lifetime);
    }
}
