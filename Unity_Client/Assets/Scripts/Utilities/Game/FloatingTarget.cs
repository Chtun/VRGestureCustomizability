using UnityEngine;

public class FloatingTarget : MonoBehaviour
{
    private TargetChallengeManager challengeManager;
    private string requiredSpell;

    public void Initialize(TargetChallengeManager manager, string spellType)
    {
        challengeManager = manager;
        requiredSpell = spellType;
    }

    void OnCollisionEnter(Collision collision)
    {
        string projectileTag = collision.gameObject.tag;

        // Check if correct spell hit
        if ((requiredSpell == "Fire" && projectileTag == "Fireball") ||
            (requiredSpell == "Lightning" && projectileTag == "Lightning") ||
            (requiredSpell == "Ice" && projectileTag == "Ice"))
        {
            Debug.Log($"{requiredSpell} target hit correctly by {projectileTag}!");
            challengeManager.TargetDestroyed(gameObject);
            Destroy(gameObject);
        }
        else
        {
            Debug.Log($"Incorrect hit: {projectileTag} tried to hit {requiredSpell} target.");
        }
    }
}
