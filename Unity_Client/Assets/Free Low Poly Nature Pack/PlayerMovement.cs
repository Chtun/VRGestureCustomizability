using UnityEngine;
using UnityEngine.InputSystem; // new input system

public class PlayerMovement : MonoBehaviour
{
    public float speed = 10f;
    public float mouseSensitivity = 150f;
    private float rotationX = 0f;
    private Transform playerCamera;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
        playerCamera = Camera.main.transform;
    }

    void Update()
    {
        // --- Look ---
        Vector2 mouseDelta = Mouse.current.delta.ReadValue();
        float mouseX = mouseDelta.x * mouseSensitivity * Time.deltaTime;
        float mouseY = mouseDelta.y * mouseSensitivity * Time.deltaTime;

        rotationX -= mouseY;
        rotationX = Mathf.Clamp(rotationX, -80f, 80f);

        transform.Rotate(Vector3.up * mouseX);
        playerCamera.localRotation = Quaternion.Euler(rotationX, 0f, 0f);

        // --- Move ---
        Vector2 moveInput = Vector2.zero;
        if (Keyboard.current.wKey.isPressed) moveInput.y += 1;
        if (Keyboard.current.sKey.isPressed) moveInput.y -= 1;
        if (Keyboard.current.aKey.isPressed) moveInput.x -= 1;
        if (Keyboard.current.dKey.isPressed) moveInput.x += 1;

        Vector3 move = (transform.right * moveInput.x + transform.forward * moveInput.y).normalized;
        transform.position += move * speed * Time.deltaTime;
    }
}
