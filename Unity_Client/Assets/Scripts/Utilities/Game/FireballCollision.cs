using UnityEngine;

public class FireballCollision : MonoBehaviour
{
    public GameObject impactEffectPrefab;
    public float damage = 10f;

    void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"Impact triggered with {collision.gameObject.name}");

        // Spawn explosion effect
        if (collision.contacts.Length > 0)
        {
            Debug.Log(impactEffectPrefab);
            // Debug.Log($"Impact effect instantiated at {collision.contacts[0].point}");
            Instantiate(
                impactEffectPrefab,
                collision.contacts[0].point,
                Quaternion.identity
            );
        }

        // Destroy fireball immediately after impact
        Destroy(gameObject);
    }
}
