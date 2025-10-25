using UnityEngine;

public class AutoDestroyAfterEffect : MonoBehaviour
{
	private ParticleSystem ps;

	void Start()
	{
		ps = GetComponent<ParticleSystem>();
		if (ps == null)
		{
			// If the particle system is not on this object, try to find it in children
			ps = GetComponentInChildren<ParticleSystem>();
		}
	}

	void Update()
	{
		if (ps != null && !ps.IsAlive())
		{
			Destroy(gameObject);
		}
	}
}
