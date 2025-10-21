using UnityEngine;

public class FireballCollision : MonoBehaviour
{
    public GameObject impactEffectPrefab;
    public float damage = 10f;

    void OnCollisionEnter(Collision collision)
    {
        // Spawn the impact effect
        if (impactEffectPrefab)
        {
            Instantiate(impactEffectPrefab, collision.contacts[0].point, Quaternion.identity);
        }

        // Destroy the fireball after collision
        Destroy(gameObject);
    }
}
