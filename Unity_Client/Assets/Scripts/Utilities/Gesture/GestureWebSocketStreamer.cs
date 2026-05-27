using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class GestureWebSocketStreamer : MonoBehaviour
{
	[Header("References")]
	[SerializeField] private JointDataGather jointDataGather;

	[Header("WebSocket Settings")]
	[SerializeField] private float sendInterval = 0.1f;

	[Header("API Settings (auto-loaded from config)")]
	private string serverUrl = "ws://127.0.0.1:8000";

	private ClientWebSocket ws;
	[SerializeField] private bool isStreaming = false;
	private CancellationTokenSource cts;

	// Add an Action to forward received messages
	public Action<string> OnGestureDataReceived;

	private void Awake()
	{
		Config config = Config.LoadConfig();
		serverUrl = config.GetWSURL();
		Debug.Log($"Loaded server URL from config: {serverUrl}");
	}

	public void Connect(bool useDefaultSystem)
	{
		// Already connected? Do nothing
		if (ws != null && ws.State == WebSocketState.Open)
			return;

		if (isStreaming)
		{
			Debug.LogWarning("WebSocket already streaming, connect ignored.");
			return;
		}

		StartCoroutine(WaitForTrackingThenConnect(useDefaultSystem));
	}


	private IEnumerator WaitForTrackingThenConnect(bool useDefaultSystem)
	{
		while (!IsBothHandsTracked())
			yield return null;

		Debug.Log("Hands tracked — connecting to gesture WebSocket server...");
		ConnectWebSocket(useDefaultSystem);
	}

	public void Disconnect()
	{
		if (!isStreaming && (ws == null || ws.State != WebSocketState.Open))
			return;

		Debug.Log("Disconnecting from WebSocket...");

		isStreaming = false;
		cts?.Cancel();

		StartCoroutine(DisconnectCoroutine());
	}

	private IEnumerator DisconnectCoroutine()
	{
		if (ws != null)
		{
			if (ws.State == WebSocketState.Open)
			{
				var task = ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client disconnecting", CancellationToken.None);
				while (!task.IsCompleted)
					yield return null;

				Debug.Log("WebSocket disconnected.");
			}

			ws.Dispose();
			ws = null;
		}

		cts?.Dispose();
		cts = null;
	}


	private bool IsBothHandsTracked()
	{
		return jointDataGather != null &&
			   jointDataGather.GetJointData(false) != null &&
			   jointDataGather.GetJointData(true) != null;
	}

	private async void ConnectWebSocket(bool useDefaultSystem)
	{
		ws = new ClientWebSocket();
		cts = new CancellationTokenSource();

		try
		{
			Uri uri = new Uri(serverUrl);
			await ws.ConnectAsync(uri, cts.Token);
			Debug.Log("Connected to gesture WebSocket server.");
			isStreaming = true;


			// Send initial system info
			var initMessage = new WebsocketInitMessage
			{
				type = "init",
				useDefaultSystem = useDefaultSystem
			};
			string jsonInit = JsonUtility.ToJson(initMessage);
			var bytes = Encoding.UTF8.GetBytes(jsonInit);
			await ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cts.Token);

			StartCoroutine(SendJointDataLoop());
			ReceiveLoop(); // Start listening for messages
		}
		catch (Exception ex)
		{
			Debug.LogError($"WebSocket connection error: {ex}");
		}
	}

	private async void ReceiveLoop()
	{
		var buffer = new byte[1024];

		try
		{
			while (ws != null && (ws.State == WebSocketState.Open || ws.State == WebSocketState.CloseReceived))
			{
				WebSocketReceiveResult result = null;

				try
				{
					result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
				}
				catch (WebSocketException ex)
				{
					Debug.LogError($"WebSocket receive error: {ex}");
					break;
				}
				catch (OperationCanceledException)
				{
					Debug.Log("WebSocket receive canceled.");
					break;
				}

				if (result == null)
					break;

				if (result.MessageType == WebSocketMessageType.Close)
				{
					Debug.LogWarning("WebSocket closed by server.");
					if (ws.State == WebSocketState.Open || ws.State == WebSocketState.CloseReceived)
					{
						await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
					}
					isStreaming = false;
					break;
				}
				else
				{
					string message = Encoding.UTF8.GetString(buffer, 0, result.Count);

					// Print the message
					Debug.Log($"Received from server: {message}");

					// Pass it along via Action if assigned
					OnGestureDataReceived?.Invoke(message);
				}
			}
		}
		finally
		{
			isStreaming = false;
		}
	}

	private IEnumerator SendJointDataLoop()
	{
		while (isStreaming)
		{
			try
			{
				var leftJoints = jointDataGather.GetJointData(false);
				var rightJoints = jointDataGather.GetJointData(true);
				var leftRoot = jointDataGather.GetRootPose(false);
				var rightRoot = jointDataGather.GetRootPose(true);

				// Create dictionaries keyed by joint ID
				var leftHandDict = new Dictionary<string, List<float>>();
				foreach (var kvp in leftJoints)
				{
					var pos = kvp.Value.position;
					leftHandDict[kvp.Key.ToString()] = new List<float> { pos.x, pos.y, pos.z };
				}

				var rightHandDict = new Dictionary<string, List<float>>();
				foreach (var kvp in rightJoints)
				{
					var pos = kvp.Value.position;
					rightHandDict[kvp.Key.ToString()] = new List<float> { pos.x, pos.y, pos.z };
				}

				// Root/wrist positions as before
				var leftWristList = new List<float> { leftRoot.position.x, leftRoot.position.y, leftRoot.position.z };
				var rightWristList = new List<float> { rightRoot.position.x, rightRoot.position.y, rightRoot.position.z };

				// Build payload with joint IDs preserved
				var payload = new
				{
					left_hand = leftHandDict,
					right_hand = rightHandDict,
					left_wrist = leftWristList,
					right_wrist = rightWristList
				};

				// Serialize to JSON string
				string json = JsonConvert.SerializeObject(payload);

				if (ws.State == WebSocketState.Open)
				{
					byte[] bytes = Encoding.UTF8.GetBytes(json);

					// Fire-and-forget SendAsync
					ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cts.Token)
						.ContinueWith(t =>
						{
							if (t.Exception != null)
								Debug.LogError($"WebSocket send failed: {t.Exception}");
						});
				}
			}
			catch (Exception ex)
			{
				Debug.LogError($"Error preparing gesture data: {ex}");
			}

			yield return new WaitForSeconds(sendInterval);
		}
	}


	private void OnDestroy()
	{
		Disconnect();
	}
}
