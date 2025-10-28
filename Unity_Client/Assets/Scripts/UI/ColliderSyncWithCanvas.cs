using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
[RequireComponent(typeof(Rigidbody))]
public class ColliderSyncWithCanvas : MonoBehaviour
{
	[Tooltip("The RectTransform to match the collider size to. Usually your Canvas or UI Panel.")]
	[SerializeField] private RectTransform targetRect;

	private BoxCollider boxCollider;

	private void Awake()
	{
		boxCollider = GetComponent<BoxCollider>();

		if (targetRect == null)
		{
			targetRect = GetComponent<RectTransform>();
		}

		// Make sure Rigidbody is kinematic if this is a UI canvas
		Rigidbody rb = GetComponent<Rigidbody>();
		rb.isKinematic = true;
	}

	private void LateUpdate()
	{
		if (targetRect == null) return;

		// Set collider size to match RectTransform
		Vector3 size = new Vector3(targetRect.rect.width, targetRect.rect.height, 0.01f);
		boxCollider.size = size;

		// Match collider position to RectTransform center
		boxCollider.center = targetRect.localPosition;
	}
}
