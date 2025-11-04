using Oculus.Interaction.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class JointDataGather : MonoBehaviour
{
	[SerializeField] private Hand leftHand;
	[SerializeField] private Hand rightHand;


	[SerializeField] private float timeBetweenSamples = 0.1f;
	[SerializeField] private string outputCSVName;
	[SerializeField] private static string liveOutputCSVPrefix = "live_recordings";
	[SerializeField] private bool isRecording = false;

	public bool IsRecording => isRecording;
	public float TimeBetweenSamples => timeBetweenSamples;

	[Header("Hand Search Settings")]
	[SerializeField] private string leftHandName = "LeftInteractions";
	[SerializeField] private string rightHandName = "RightInteractions";


	private string scriptName = "JointDataGather";

	private void OnEnable()
	{
		// Subscribe to scene load events
		SceneManager.sceneLoaded += OnSceneLoaded;

		// Also refresh hands immediately in case we're already in a scene
		RefreshHandReferences();
	}

	private void OnDisable()
	{
		// Unsubscribe to avoid memory leaks
		SceneManager.sceneLoaded -= OnSceneLoaded;
	}

	private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
	{
		RefreshHandReferences();
	}

	private void RefreshHandReferences()
	{
		// Find left hand
		if (leftHand == null)
		{
			Transform leftTransform = GameObject.Find(leftHandName).transform;
			if (leftTransform != null)
				leftHand = leftTransform.GetComponent<Hand>();
			else
				Debug.LogError($"[{scriptName}] Left hand '{leftHandName}' not found!");
		}

		// Find right hand
		if (rightHand == null)
		{
			Transform rightTransform = GameObject.Find(rightHandName).transform;
			if (rightTransform != null)
				rightHand = rightTransform.GetComponent<Hand>();
			else
				Debug.LogError($"[{scriptName}] Right hand '{rightHandName}' not found!");
		}
	}

	private void StartRecording(string label = null)
	{
		string outputPath;
		if (string.IsNullOrEmpty(label))
		{
			outputPath = Path.Combine(Application.persistentDataPath, $"{outputCSVName}.csv");
		}
		else
		{
			outputCSVName = GetRecordedCSVName(label);
			outputPath = GetRecordedCSVPath(label);

			// Clean out the current file if it exists
			if (File.Exists(outputPath))
			{
				string[] lines = File.ReadAllLines(outputPath);
				if (lines.Length > 0)
				{
					// Keep only the first line (header)
					File.WriteAllText(outputPath, lines[0] + Environment.NewLine);
					Debug.Log($"[{scriptName}] Cleared file but kept header: {outputPath}");
				}
				else
				{
					Debug.Log($"[{scriptName}] File was empty: {outputPath}");
				}
			}
		}


		Debug.Log("Recording to CSV path: " + outputPath);

		if (!isRecording)
		{
			isRecording = true;
			StartCoroutine(RecordDataLoop(outputPath));
		}
	}

	public void StopRecording()
	{
		isRecording = false;
	}

	public static string GetRecordedCSVPath(string label)
	{
		return Path.Combine(Application.persistentDataPath, $"{GetRecordedCSVName(label)}.csv");
	}

	public static string GetRecordedCSVName(string label)
	{
		return $"{liveOutputCSVPrefix}-{label}";
	}

	private IEnumerator RecordDataLoop(string outputPath)
	{
		while (isRecording)
		{
			RecordData(outputPath);
			yield return new WaitForSeconds(timeBetweenSamples); // repeat after the specified time between samples.
		}
	}

	public static List<HandJointId> ImportantHandJointIDs()
	{
		return new List<HandJointId> {
			HandJointId.HandThumbTip,
			HandJointId.HandThumb3,
			HandJointId.HandThumb2,
			HandJointId.HandThumb1,
			HandJointId.HandRingTip,
			HandJointId.HandRing3,
			HandJointId.HandRing2,
			HandJointId.HandRing1,
			HandJointId.HandRing0,
			HandJointId.HandPinkyTip,
			HandJointId.HandPinky3,
			HandJointId.HandPinky2,
			HandJointId.HandPinky1,
			HandJointId.HandPinky0,
			HandJointId.HandMiddleTip,
			HandJointId.HandMiddle3,
			HandJointId.HandMiddle2,
			HandJointId.HandMiddle1,
			HandJointId.HandMiddle0,
			HandJointId.HandIndexTip,
			HandJointId.HandIndex3,
			HandJointId.HandIndex2,
			HandJointId.HandIndex1,
			HandJointId.HandIndex0,
		};
	}

	private static bool IsDataReliable(Hand hand)
	{
		return hand.IsHighConfidence;
	}

	public bool IsDataReliable(bool isRightHand)
	{
		if (isRightHand)
		{
			return IsDataReliable(this.rightHand);
		}
		else
		{
			return IsDataReliable(this.leftHand);
		}
	}

	private static Dictionary<HandJointId, Pose> GetJointData(Hand hand)
	{
		Dictionary<HandJointId, Pose> currentJointPoses = new Dictionary<HandJointId, Pose>();

		foreach (HandJointId handJointId in ImportantHandJointIDs())
		{
			try
			{
				Pose currentPose;
				hand.GetJointPoseFromWrist(handJointId, out currentPose);

				currentJointPoses[handJointId] = currentPose;
			}
			catch (Exception e)
			{
				Debug.LogException(e);
				currentJointPoses[handJointId] = default;
			}

		}

		return currentJointPoses;
	}

	public Dictionary<HandJointId, Pose> GetJointData(bool isRightHand)
	{
		if (isRightHand)
		{
			return GetJointData(this.rightHand);
		}
		else
		{
			return GetJointData(this.leftHand);
		}
	}

	public Pose GetRootPose(bool isRightHand)
	{
		Pose rootPose;
		if (isRightHand)
		{
			this.rightHand.GetRootPose(out rootPose);
		}
		else
		{
			this.leftHand.GetRootPose(out rootPose);
		}

		return rootPose;
	}



	private bool WriteData(
	Dictionary<HandJointId, Pose> leftJointPoses,
	Dictionary<HandJointId, Pose> rightJointPoses,
	Pose leftRootPose,
	Pose rightRootPose,
	string outputPath)
	{
		try
		{
			// Only write header if file does NOT exist
			if (!File.Exists(outputPath))
			{
				using (StreamWriter writer = new StreamWriter(outputPath, false))
				{
					List<string> headerColumns = new List<string>();
					headerColumns.Add("Timestamp");

					// Left hand joints
					foreach (HandJointId handJointId in leftJointPoses.Keys)
					{
						headerColumns.Add($"L_{handJointId}_posX");
						headerColumns.Add($"L_{handJointId}_posY");
						headerColumns.Add($"L_{handJointId}_posZ");
						headerColumns.Add($"L_{handJointId}_rotX");
						headerColumns.Add($"L_{handJointId}_rotY");
						headerColumns.Add($"L_{handJointId}_rotZ");
						headerColumns.Add($"L_{handJointId}_rotW");
					}

					// Right hand joints
					foreach (HandJointId handJointId in rightJointPoses.Keys)
					{
						headerColumns.Add($"R_{handJointId}_posX");
						headerColumns.Add($"R_{handJointId}_posY");
						headerColumns.Add($"R_{handJointId}_posZ");
						headerColumns.Add($"R_{handJointId}_rotX");
						headerColumns.Add($"R_{handJointId}_rotY");
						headerColumns.Add($"R_{handJointId}_rotZ");
						headerColumns.Add($"R_{handJointId}_rotW");
					}

					// Root poses
					headerColumns.AddRange(new string[] {
					"L_Root_posX","L_Root_posY","L_Root_posZ","L_Root_rotX","L_Root_rotY","L_Root_rotZ","L_Root_rotW",
					"R_Root_posX","R_Root_posY","R_Root_posZ","R_Root_rotX","R_Root_rotY","R_Root_rotZ","R_Root_rotW"
				});

					writer.WriteLine(string.Join(",", headerColumns));
				}
			}

			// Append a new row of data
			using (StreamWriter writer = new StreamWriter(outputPath, true))
			{
				List<string> rowData = new List<string>
			{
				Time.time.ToString() // Timestamp
            };

				// Left joints
				foreach (var jointPose in leftJointPoses.Values)
				{
					rowData.Add(jointPose.position.x.ToString());
					rowData.Add(jointPose.position.y.ToString());
					rowData.Add(jointPose.position.z.ToString());
					rowData.Add(jointPose.rotation.x.ToString());
					rowData.Add(jointPose.rotation.y.ToString());
					rowData.Add(jointPose.rotation.z.ToString());
					rowData.Add(jointPose.rotation.w.ToString());
				}

				// Right joints
				foreach (var jointPose in rightJointPoses.Values)
				{
					rowData.Add(jointPose.position.x.ToString());
					rowData.Add(jointPose.position.y.ToString());
					rowData.Add(jointPose.position.z.ToString());
					rowData.Add(jointPose.rotation.x.ToString());
					rowData.Add(jointPose.rotation.y.ToString());
					rowData.Add(jointPose.rotation.z.ToString());
					rowData.Add(jointPose.rotation.w.ToString());
				}

				// Root poses
				rowData.Add(leftRootPose.position.x.ToString());
				rowData.Add(leftRootPose.position.y.ToString());
				rowData.Add(leftRootPose.position.z.ToString());
				rowData.Add(leftRootPose.rotation.x.ToString());
				rowData.Add(leftRootPose.rotation.y.ToString());
				rowData.Add(leftRootPose.rotation.z.ToString());
				rowData.Add(leftRootPose.rotation.w.ToString());

				rowData.Add(rightRootPose.position.x.ToString());
				rowData.Add(rightRootPose.position.y.ToString());
				rowData.Add(rightRootPose.position.z.ToString());
				rowData.Add(rightRootPose.rotation.x.ToString());
				rowData.Add(rightRootPose.rotation.y.ToString());
				rowData.Add(rightRootPose.rotation.z.ToString());
				rowData.Add(rightRootPose.rotation.w.ToString());

				writer.WriteLine(string.Join(",", rowData));
			}

			return true;
		}
		catch (Exception e)
		{
			Debug.LogError($"Failed to write CSV: {e}");
			return false;
		}
	}


	private void RecordData(string outputPath)
	{
		// Collect joint data
		Dictionary<HandJointId, Pose> currentLeftJointPoses = GetJointData(leftHand);
		Dictionary<HandJointId, Pose> currentRightJointPoses = GetJointData(rightHand);

		// Collect root poses
		Pose currentRightRootPose;
		Pose currentLeftRootPose;
		rightHand.GetRootPose(out currentRightRootPose);
		leftHand.GetRootPose(out currentLeftRootPose);

		// Write to CSV
		WriteData(currentLeftJointPoses, currentRightJointPoses, currentLeftRootPose, currentRightRootPose, outputPath);
	}

	public IEnumerator WaitForHandsThenRecord(string label = null)
	{
		if (leftHand == null || rightHand == null)
		{
			Debug.LogError($"[{scriptName}] The left or right hand does not have a proper reference!");
			RefreshHandReferences();
			yield return null;
		}

		// Wait until both hands are tracked
		while (!IsDataReliable(isRightHand: false) || !IsDataReliable(isRightHand: true))
		{
			Debug.Log("Both hands are not tracked yet!");
			yield return null; // wait for next frame
		}

		Debug.Log($"[{scriptName}] Both hands are tracked. Starting recording.");
		StartRecording(label);
	}

	public static GestureInput ReadCSVToGestureInput(string filePath, string gestureLabel)
	{
		if (!File.Exists(filePath))
		{
			Debug.LogError($"CSV file not found: {filePath}");
			return null;
		}

		var lines = File.ReadAllLines(filePath);
		if (lines.Length < 2)
		{
			Debug.LogError("CSV file does not contain enough data.");
			return null;
		}

		string[] headers = lines[0].Split(',');

		List<string> leftJointNames = new List<string>();
		List<string> rightJointNames = new List<string>();

		foreach (var header in headers)
		{
			if (header.StartsWith("L_") && header.Contains("_pos") && !header.Contains("Root"))
			{
				string jointName = header.Substring(2, header.IndexOf("_pos") - 2);
				if (!leftJointNames.Contains(jointName))
					leftJointNames.Add(jointName);
			}
			else if (header.StartsWith("R_") && header.Contains("_pos") && !header.Contains("Root"))
			{
				string jointName = header.Substring(2, header.IndexOf("_pos") - 2);
				if (!rightJointNames.Contains(jointName))
					rightJointNames.Add(jointName);
			}
		}

		int numLeftJoints = leftJointNames.Count;
		int numRightJoints = rightJointNames.Count;

		if (numLeftJoints != 24 || numRightJoints != 24)
		{
			Debug.LogError("The number of left joints or right joint names in the header of the file do not match! Please ensure file header is correct!");
			return null;
		}

		var leftJointPositions = new List<List<List<float>>>();
		var rightJointPositions = new List<List<List<float>>>();
		var leftJointRotations = new List<List<List<float>>>();
		var rightJointRotations = new List<List<List<float>>>();
		var leftWristPositions = new List<List<float>>();
		var rightWristPositions = new List<List<float>>();
		var leftWristRotations = new List<List<float>>();
		var rightWristRotations = new List<List<float>>();

		for (int i = 1; i < lines.Length; i++)
		{
			string[] cols = lines[i].Split(',');
			int colIndex = 1; // skip Timestamp

			// LEFT hand
			var leftFramePos = new List<List<float>>();
			var leftFrameRot = new List<List<float>>();
			for (int j = 0; j < numLeftJoints; j++)
			{
				// position (3)
				var jointPos = new List<float>
			{
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
			};

				// rotation (4)
				var jointRot = new List<float>
			{
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
			};

				leftFramePos.Add(jointPos);
				leftFrameRot.Add(jointRot);
			}
			leftJointPositions.Add(leftFramePos);
			leftJointRotations.Add(leftFrameRot);

			// RIGHT hand
			var rightFramePos = new List<List<float>>();
			var rightFrameRot = new List<List<float>>();
			for (int j = 0; j < numRightJoints; j++)
			{
				var jointPos = new List<float>
			{
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
			};

				var jointRot = new List<float>
			{
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
				float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
			};

				rightFramePos.Add(jointPos);
				rightFrameRot.Add(jointRot);
			}
			rightJointPositions.Add(rightFramePos);
			rightJointRotations.Add(rightFrameRot);

			// LEFT wrist (root)
			var leftRootPos = new List<float>
		{
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
		};
			var leftRootRot = new List<float>
		{
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
		};
			leftWristPositions.Add(leftRootPos);
			leftWristRotations.Add(leftRootRot);

			// RIGHT wrist (root)
			var rightRootPos = new List<float>
		{
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
		};
			var rightRootRot = new List<float>
		{
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture),
			float.Parse(cols[colIndex++], CultureInfo.InvariantCulture)
		};
			rightWristPositions.Add(rightRootPos);
			rightWristRotations.Add(rightRootRot);
		}

		return new GestureInput(
			gestureLabel,
			leftJointPositions,
			rightJointPositions,
			leftJointRotations,
			rightJointRotations,
			leftWristPositions,
			rightWristPositions,
			 leftWristRotations,
			 rightWristRotations
		);
	}




	// To stop:
	void OnDestroy()
	{
		StopRecording();
	}

}