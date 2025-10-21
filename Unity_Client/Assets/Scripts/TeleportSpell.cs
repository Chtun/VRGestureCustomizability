using UnityEngine;

public class TeleportController : MonoBehaviour
{
    public Transform player;                   // PLAYER object
    public Transform cameraTransform;          // Main Camera
    public GameObject teleportIndicator;       // The visible target marker

    public LayerMask teleportableLayers = ~0;  // Layers you can teleport onto (default: everything)
    public KeyCode aimKey = KeyCode.T;         // Key to show/aim teleport
    public KeyCode teleportKey = KeyCode.Y;    // Key to teleport
    public float indicatorHeightOffset = 0.01f; // Slight offset so indicator doesn't clip into floor

    private Vector3 targetPosition;
    private bool validTarget = false;

    void Start()
    {
        if (teleportIndicator != null)
            teleportIndicator.SetActive(false); // Hide indicator at start
    }

    void Update()
    {
        if (Input.GetKeyDown(aimKey))
            Debug.Log("Aim key pressed!");
        if (Input.GetKeyDown(teleportKey))
            Debug.Log("Teleport key pressed!");

        HandleTeleportAiming();
        HandleTeleportActivation();
    }

    private bool targetLocked = false; // new: lock the target while aiming

    void HandleTeleportAiming()
    {
        if (Input.GetKey(aimKey))
        {
            Ray ray = new Ray(cameraTransform.position, cameraTransform.forward);
            if (Physics.Raycast(ray, out RaycastHit hit, 100f, teleportableLayers))
            {
                targetPosition = hit.point;
                targetLocked = true;      // lock target
                validTarget = true;

                if (teleportIndicator != null)
                {
                    teleportIndicator.SetActive(true);
                    teleportIndicator.transform.position = hit.point + Vector3.up * indicatorHeightOffset;
                }
            }
            else
            {
                validTarget = false;
                if (teleportIndicator != null)
                    teleportIndicator.SetActive(false);
            }
        }
        else if (!targetLocked)
        {
            if (teleportIndicator != null)
                teleportIndicator.SetActive(false);
        }
    }

    void HandleTeleportActivation()
    {
        if (Input.GetKeyDown(teleportKey) && validTarget)
        {
            Debug.Log($"Teleporting to {targetPosition}");
            player.position = targetPosition + Vector3.up * 1f;
            validTarget = false;
            targetLocked = false;

            if (teleportIndicator != null)
                teleportIndicator.SetActive(false);
        }
    }

}
