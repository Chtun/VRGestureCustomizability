using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[Serializable]
public class Config
{
	public string host = "127.0.0.1";
	public string port = "8000";

	// Dictionary to hold default gestures
	public Dictionary<string, ActionType> defaultGestures = new Dictionary<string, ActionType>();

	public string GetHTTPURL() => $"http://{host}:{port}";
	public string GetWSURL() => $"ws://{host}:{port}/ws";

	public static Config LoadConfig()
	{
		string path = Path.Combine(Application.streamingAssetsPath, "config.json");

		try
		{
			if (!File.Exists(path))
			{
				Debug.LogWarning($"config.json not found at: {path}, using default values.");
				return new Config();
			}

			string json = File.ReadAllText(path);

			// Parse basic config
			Config cfg = JsonUtility.FromJson<Config>(json);

			// Parse defaultGestures separately using Newtonsoft.Json
			JObject obj = JObject.Parse(json);
			var defaultsToken = obj["defaultGestures"];
			if (defaultsToken != null)
			{
				foreach (var prop in defaultsToken.Children<JProperty>())
				{
					string gestureName = prop.Name;
					string actionTypeStr = prop.Value.ToString();

					if (Enum.TryParse<ActionType>(actionTypeStr, out ActionType action))
					{
						cfg.defaultGestures[gestureName] = action;
					}
					else
					{
						Debug.LogWarning($"Invalid ActionType '{actionTypeStr}' for gesture '{gestureName}' in config.");
					}
				}
			}

			return cfg;
		}
		catch (Exception ex)
		{
			Debug.LogError($"Error loading config: {ex}");
			return new Config();
		}
	}
}
