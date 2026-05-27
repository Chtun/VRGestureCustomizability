using UnityEngine;
using UnityEngine.InputSystem;

public class TeleportController : MonoBehaviour
{
    [Header("References")]
    public Transform player;
    public Transform cameraTransform;
    public GameObject teleportIndicator;
    public InputManager inputManager;

    [Header("Settings")]
    public LayerMask teleportableLayers = ~0;
    public float indicatorHeightOffset = 0.01f;
    public float smoothMoveSpeed = 10f;
    public float teleportHeightOffset = 1f;

    private Vector3 targetPosition;
    private bool validTarget = false;
    private bool isAiming = false;

    void Awake()
    {
        if (inputManager == null)
        {
            inputManager = FindFirstObjectByType<InputManager>();
            if (inputManager == null)
            {
                Debug.LogError("TeleportController: No InputManager found in scene!");
                enabled = false;
                return;
            }
        }

        if (teleportIndicator != null)
            teleportIndicator.SetActive(false);
    }

    void OnEnable()
    {
        inputManager.OnTeleportAim += ToggleAiming;
        inputManager.OnTeleportCast += HandleTeleportActivation;
    }

    void OnDisable()
    {
        inputManager.OnTeleportAim -= ToggleAiming;
        inputManager.OnTeleportCast -= HandleTeleportActivation;
    }

    void Update()
    {
        if (isAiming)
            HandleTeleportAiming();
    }

    private void ToggleAiming()
    {
        // Toggle on/off each time the aim key is pressed
        isAiming = !isAiming;
        if (!isAiming)
        {
            if (teleportIndicator != null)
                teleportIndicator.SetActive(false);
            validTarget = false;
        }
    }

    private void HandleTeleportAiming()
    {
        if (!isAiming)
            return;

        // Ray from the camera forward direction
        Ray ray = new Ray(cameraTransform.position, cameraTransform.forward);

        if (Physics.Raycast(ray, out RaycastHit hit, 100f, teleportableLayers))
        {
            validTarget = true;
            targetPosition = hit.point;

            if (teleportIndicator != null)
            {
                teleportIndicator.SetActive(true);

                // Smooth follow for stability
                teleportIndicator.transform.position = Vector3.Lerp(
                    teleportIndicator.transform.position,
                    hit.point + Vector3.up * indicatorHeightOffset,
                    Time.deltaTime * smoothMoveSpeed
                );

                // Always face up, ignore surface rotation
                teleportIndicator.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            }
        }
        else
        {
            validTarget = false;
            if (teleportIndicator != null)
                teleportIndicator.SetActive(false);
        }
    }

    private void HandleTeleportActivation()
    {
        if (validTarget)
        {
            Debug.Log($"Teleporting to {targetPosition}");
            player.position = targetPosition + Vector3.up * teleportHeightOffset;

            // Reset state
            validTarget = false;
            isAiming = false;
            if (teleportIndicator != null)
                teleportIndicator.SetActive(false);
        }
    }
}
