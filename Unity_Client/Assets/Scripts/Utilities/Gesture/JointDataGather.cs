using Oculus.Interaction.Input;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class JointDataGather : MonoBehaviour
{
	[SerializeField] private Hand leftHand;
	[SerializeField] private Hand rightHand;


	[SerializeField] private float timeBetweenSamples = 0.1f;
	[SerializeField] private string outputCSVName;
	[SerializeField] private bool isRecording = false;

	public void StartRecording()
	{
		if (!isRecording)
		{
			isRecording = true;
			StartCoroutine(RecordDataLoop());
		}
	}

	public void StopRecording()
	{
		isRecording = false;
	}

	private IEnumerator RecordDataLoop()
	{
		while (isRecording)
		{
			RecordData();
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


	private void RecordData()
	{
		string outputPath = Path.Combine(Application.persistentDataPath, $"{outputCSVName}.csv");
		Debug.Log("CSV path: " + outputPath);


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


	void Start()
	{
		// Start a coroutine that waits for tracking before recording
		StartCoroutine(WaitForHandsThenRecord());
	}

	private IEnumerator WaitForHandsThenRecord()
	{
		// Wait until both hands are tracked
		while (!leftHand.IsTrackedDataValid || !rightHand.IsTrackedDataValid)
		{
			yield return null; // wait for next frame
		}

		Debug.Log("Both hands are tracked. Starting recording.");
		StartRecording();
	}


	// To stop:
	void OnDestroy()
	{
		StopRecording();
	}

}