using UnityEngine;
using UnityEngine.InputSystem;

public class InputManager : MonoBehaviour
{

	public event System.Action OnTeleportAim;
	public event System.Action OnTeleportCast;
	public event System.Action OnFireballCast;
	public event System.Action OnLightningCast;


	void Update()
	{

		// Check keyboard inputs
		if (Keyboard.current == null) return;

		if (Keyboard.current.tKey.wasPressedThisFrame)
		{
			OnTeleportAim?.Invoke();
		}

		if (Keyboard.current.yKey.wasPressedThisFrame)
		{
			OnTeleportCast?.Invoke();
		}

		if (Keyboard.current.fKey.wasPressedThisFrame)
		{
			OnFireballCast?.Invoke();
		}

		if (Keyboard.current.lKey.wasPressedThisFrame)
        {
            OnLightningCast?.Invoke();
        }
	}

	public void TakeAction(ActionType actionType)
	{
		switch (actionType)
		{
			case ActionType.CastFireball:
				CastFireball(); break;
			case ActionType.CastLightning:
				CastLightning(); break;
			case ActionType.CastTeleport:
				CastTeleport(); break;
			case ActionType.ShowTeleport:
				ShowTeleport(); break;
		}
	}

	public void CastFireball()
	{
		OnFireballCast?.Invoke();
	}

	public void ShowTeleport()
	{
		OnTeleportAim.Invoke();
	}

	public void CastTeleport()
	{
		OnTeleportCast?.Invoke();
	}

	public void CastLightning()
    {
        OnLightningCast?.Invoke();
    }
}

public enum ActionType
{
	CastFireball,
	CastLightning,
	ShowTeleport,
	CastTeleport,
}