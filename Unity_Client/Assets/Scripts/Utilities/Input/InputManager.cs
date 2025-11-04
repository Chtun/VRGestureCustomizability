using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.InputSystem;

public class InputManager : MonoBehaviour
{

	public event System.Action OnTeleportAim;
	public event System.Action OnTeleportCast;
	public event System.Action OnFireballCast;
	public event System.Action OnLightningCast;
	public event System.Action OnIceCast;


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

		if (Keyboard.current.iKey.wasPressedThisFrame)
		{
			OnIceCast?.Invoke();
		}
	}

	public void TakeAction(ActionType actionType)
	{
		Debug.Log($"Taking action: {actionType.ToString()}");
		switch (actionType)
		{
			case ActionType.CastFireball:
				CastFireball(); break;
			case ActionType.CastLightning:
				CastLightning(); break;
			case ActionType.CastIce:
				CastIce(); break;
			case ActionType.CastTeleport:
				CastTeleport(); break;
			case ActionType.ShowTeleport:
				ShowTeleport(); break;
			case ActionType.Inactivated:
				break;
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

	public static string ActionTypeName(ActionType actionType)
	{
		// Convert enum to string
		string name = actionType.ToString();

		// Insert space between lowercase and uppercase letters
		string formattedName = Regex.Replace(name, "([a-z])([A-Z])", "$1 $2");

		return formattedName;
	}

	public void CastLightning()
    {
        OnLightningCast?.Invoke();
    }

	public void CastIce()
	{
		OnIceCast?.Invoke();
	}
}

public enum ActionType
{
	CastFireball,
	CastLightning,
	CastIce,
	ShowTeleport,
	CastTeleport,

	Inactivated,
}