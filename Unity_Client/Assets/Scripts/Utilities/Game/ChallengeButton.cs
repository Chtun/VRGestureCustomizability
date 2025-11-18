using UnityEngine;
using UnityEngine.InputSystem;
using TMPro;

public class ChallengeButton : MonoBehaviour
{
    public TargetChallengeManager challengeManager;
    public KeyCode testKey = KeyCode.B;
    public TextMeshProUGUI label;
    private bool challengeActive = false;
    private Renderer buttonRenderer;

    private void Start()
    {
        buttonRenderer = GetComponentInChildren<Renderer>();
        if (label != null)
            label.text = "Press button to start challenge!";
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.bKey.wasPressedThisFrame)
        {
            PressButton();
        }
    }

    public void PressButton()
    {
        Debug.Log("Button has been pressed!");

        if (challengeActive) return;

        challengeActive = true;
        if (buttonRenderer) buttonRenderer.material.color = Color.red;
        if (label != null) label.text = "Challenge in progress...";
        challengeManager.StartChallenge();
    }

    public void ResetButton()
    {
        challengeActive = false;
        if (buttonRenderer) buttonRenderer.material.color = Color.green;
        if (label != null) label.text = "Press button to start challenge!";
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Hand"))
        {
            PressButton();
        }

        Debug.Log("Object collided with button");
    }


}
