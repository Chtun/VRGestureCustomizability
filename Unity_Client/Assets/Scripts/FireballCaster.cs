using UnityEngine;

public class FireballCaster : MonoBehaviour
{
    public GameObject fireballPrefab;
    public GameObject impactEffectPrefab;
    public Transform spawnPoint;
    public float fireballSpeed = 25f;
    public float lifetime = 5f;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.F))
        {
            CastFireball();
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

        Rigidbody rb = fireball.GetComponent<Rigidbody>();
        if (rb != null)
        {
            // Get forward direction from camera
            Transform cameraTransform = Camera.main.transform;
            Vector3 shootDirection = cameraTransform.forward;

            rb.linearVelocity = shootDirection * fireballSpeed;
        }

        var particleSystem = fireball.GetComponentInChildren<ParticleSystem>();
        //if (particleSystem != null)
        //{
        //    var main = particleSystem.main;
        //    main.startColor = Color.red;
        //}

        // Optionally auto-destroy after 5 seconds
        Destroy(fireball, 5f);
    }
}
