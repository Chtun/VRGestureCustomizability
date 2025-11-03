using UnityEngine;
using UnityEngine.InputSystem;

public class ChallengeButton : MonoBehaviour
{
    public TargetChallengeManager challengeManager;
    public KeyCode testKey = KeyCode.B; // press 'B' to start in non-VR mode

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.bKey.wasPressedThisFrame)
        {
            challengeManager.StartChallenge();
        }
    }

    // VR-friendly method to call from a collider or hand script
    public void PressButton()
    {
        challengeManager.StartChallenge();
    }

    // Optional: small animation feedback
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Hand")) // VR integration-ready
        {
            PressButton();
        }
    }
}
