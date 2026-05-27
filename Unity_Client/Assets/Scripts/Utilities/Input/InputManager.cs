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

	public bool ManualOverride = false;

	private void Awake()
	{
		Config config = Config.LoadConfig();
		ManualOverride = config.GetManualOverride();
		Debug.Log($"Loaded Manual Override from config: (Manual Override: {ManualOverride})");
	}

	void Update()
	{

		// Check keyboard inputs
		if (Keyboard.current == null) return;

		if (ManualOverride)
		{
			if (Keyboard.current.digit1Key.wasPressedThisFrame)
			{
				OnFireballCast?.Invoke();
			}

			if (Keyboard.current.digit2Key.wasPressedThisFrame)
			{
				OnLightningCast?.Invoke();
			}

			if (Keyboard.current.digit3Key.wasPressedThisFrame)
			{
				OnIceCast?.Invoke();
			}

			if (Keyboard.current.digit4Key.wasPressedThisFrame)
			{
				OnTeleportAim?.Invoke();
			}

			if (Keyboard.current.digit5Key.wasPressedThisFrame)
			{
				OnTeleportCast?.Invoke();
			}
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