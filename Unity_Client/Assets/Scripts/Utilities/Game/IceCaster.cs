using UnityEngine;

public class IceCaster : MonoBehaviour
{
    public GameObject projectilePrefab;
    public GameObject impactEffectPrefab;
    public Transform spawnPoint;
    public float projectileSpeed = 15f;
    public float lifetime = 5f;
    public float cooldown = 2f;

    private InputManager inputManager; // Reference to the Input Manager
    private float lastFireTime = -Mathf.Infinity;

    void Awake()
    {
        inputManager = FindFirstObjectByType<InputManager>();
        if (inputManager == null)
        {
            Debug.LogError("InputManager not found! IceCaster cannot subscribe to input events.");
            enabled = false;
        }
    }

    void OnEnable()
    {
        if (inputManager != null)
        {
            inputManager.OnIceCast += TryCastIce;
        }
    }

    void OnDisable()
    {
        if (inputManager != null)
        {
            inputManager.OnIceCast -= TryCastIce;
        }
    }

    private void TryCastIce()
    {
        float timeSinceLast = Time.time - lastFireTime;

        if (timeSinceLast >= cooldown)
        {
            Debug.Log($"[Ice] Casting Ice at {Time.time:F2} (cooldown met)");
            CastIce();
            lastFireTime = Time.time;
        }
        else
        {
            float remaining = cooldown - timeSinceLast;
            Debug.Log($"[Ice] Still cooling down ({remaining:F1}s remaining)");
        }
    }

    void CastIce()
    {
        if (projectilePrefab == null || spawnPoint == null)
        {
            Debug.LogWarning("Missing prefab or spawn point!");
            return;
        }

        GameObject ice = Instantiate(projectilePrefab, spawnPoint.position, spawnPoint.rotation);

        var collision = ice.GetComponent<IceCollision>();
        if (collision != null && impactEffectPrefab != null)
            collision.impactEffectPrefab = impactEffectPrefab;

        var rb = ice.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Transform cameraTransform = Camera.main.transform;
            Vector3 shootDirection = cameraTransform.forward;
            rb.linearVelocity = shootDirection * projectileSpeed;
        }

        Destroy(ice, lifetime);
    }
}
