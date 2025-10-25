using UnityEngine;
using UnityEngine.InputSystem; // Necessary for InputAction

public class TeleportController : MonoBehaviour
{
	// 1. Link to the Input Action Asset's specific Actions

	[Header("Components")]
	public Transform player;
	public Transform cameraTransform;
	public GameObject teleportIndicator;

	public InputManager inputManager;

	[Header("Settings")]
	public LayerMask teleportableLayers = ~0;
	public float indicatorHeightOffset = 0.01f;

	private Vector3 targetPosition;
	private bool validTarget = false;

	void Awake()
	{
		// Get the InputManager reference once at startup
		inputManager = FindFirstObjectByType<InputManager>();
		if (inputManager == null)
		{
			Debug.LogError("InputManager not found! TeleportController cannot function.");
			enabled = false;
		}
	}

	void OnEnable()
	{
		if (inputManager != null)
		{
			// Subscribe to the InputManager's events when the script is enabled
			inputManager.OnTeleportAim += HandleTeleportAiming;
			inputManager.OnTeleportCast += HandleTeleportActivation;
		}

		if (teleportIndicator != null)
			teleportIndicator.SetActive(false);
	}

	void OnDisable()
	{
		if (inputManager != null)
		{
			// Unsubscribe from the InputManager's events when the script is disabled
			inputManager.OnTeleportAim -= HandleTeleportAiming;
			inputManager.OnTeleportCast -= HandleTeleportActivation;
		}
	}

	// --- Action Event Handlers ---

	private void OnAimPerformed(InputAction.CallbackContext context)
	{
		// This runs the moment the Aim key is held down (Input.GetKey(aimKey) equivalent)
		// We put the aiming logic here, but it needs to run every frame, so we still use Update/LateUpdate
		Debug.Log("Aim key pressed/held!");
	}

	private void OnAimCanceled(InputAction.CallbackContext context)
	{
		// This runs the moment the Aim key is released
		if (teleportIndicator != null)
			teleportIndicator.SetActive(false);
		validTarget = false;
	}

	private void OnTeleportPerformed(InputAction.CallbackContext context)
	{
		// This runs the moment the Teleport key is pressed (Input.GetKeyDown(teleportKey) equivalent)
		HandleTeleportActivation();
		Debug.Log("Teleport key pressed!");
	}

	private bool targetLocked = false; // new: lock the target while aiming

	void HandleTeleportAiming()
	{
		if (!targetLocked)
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
				targetLocked = false;
				validTarget = false;
				if (teleportIndicator != null)
					teleportIndicator.SetActive(false);
			}
		}
		else
		{
			targetLocked = false;
			validTarget = false;
			if (teleportIndicator != null)
				teleportIndicator.SetActive(false);
		}
	}

	void HandleTeleportActivation()
	{
		if (validTarget)
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