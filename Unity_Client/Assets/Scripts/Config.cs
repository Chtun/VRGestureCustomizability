using System;
using System.IO;
using UnityEngine;

[Serializable]
public class Config
{
	public string host = "127.0.0.1";
	public string port = "8000";

	public string GetHTTPURL()
	{
		return "http://" + host + ":" + port;
	}

	public string GetWSURL()
	{
		return "ws://" + host + ":" + port + "/ws";
	}

	public static Config LoadConfig()
	{
		string path = Path.Combine(Application.streamingAssetsPath, "config.json");

		try
		{
			if (File.Exists(path))
			{
				string json = File.ReadAllText(path);
				return JsonUtility.FromJson<Config>(json);
			}
			else
			{
				Debug.LogWarning($"config.json not found under path: {path}, using default URL.");
				return new Config();
			}
		}
		catch (Exception ex)
		{
			Debug.LogError($"Error loading config: {ex}");
			return new Config();
		}
	}
}
