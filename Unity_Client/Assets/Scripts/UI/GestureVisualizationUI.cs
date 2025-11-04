using Oculus.Interaction;
using Oculus.Interaction.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GestureVisualizationUI : MonoBehaviour
{

	private BaseGestureData baseGestureData;

	[Header("Hand Visualizations")]
	[SerializeField] private HandVisual leftHandVisual;
	[SerializeField] private HandVisual rightHandVisual;

	[Header("Visualization Video Time")]
	[SerializeField] public float timeBetweenFrames = 0.1f;
	[SerializeField] private int currentFrame = 0;

	private bool isPlaying = false;
	private Coroutine playbackCoroutine;

	private Vector3 leftWristOffset;
	private Quaternion leftWristRotationOffset;
	private Vector3 rightWristOffset;
	private Quaternion rightWristRotationOffset;


	public void Initialize(BaseGestureData baseGestureData)
	{
		if (baseGestureData == null)
			return;

		this.baseGestureData = baseGestureData;

		NormalizeData();

		// Show first frame
		UpdateVisualizationFrame(0);
	}

	public void NormalizeData()
	{
		if (baseGestureData.left_wrist_positions.Count == 0)
			return;

		// --- Reference position (first frame left wrist) ---
		Vector3 referencePos = new Vector3(
			baseGestureData.left_wrist_positions[0][0],
			baseGestureData.left_wrist_positions[0][1],
			baseGestureData.left_wrist_positions[0][2]
		);

		leftWristOffset = referencePos;
		rightWristOffset = referencePos; // same reference for right hand

		int frameCount = baseGestureData.left_wrist_positions.Count;
		int jointCount = baseGestureData.left_joint_rotations[0].Count;

		for (int i = 0; i < frameCount; i++)
		{
			// --- Normalize wrist positions ---
			// Left
			Vector3 leftPos = new Vector3(
				baseGestureData.left_wrist_positions[i][0],
				baseGestureData.left_wrist_positions[i][1],
				baseGestureData.left_wrist_positions[i][2]
			);
			Vector3 normalizedLeftPos = leftPos - referencePos;
			baseGestureData.left_wrist_positions[i][0] = normalizedLeftPos.x;
			baseGestureData.left_wrist_positions[i][1] = normalizedLeftPos.y;
			baseGestureData.left_wrist_positions[i][2] = normalizedLeftPos.z;

			// Right
			Vector3 rightPos = new Vector3(
				baseGestureData.right_wrist_positions[i][0],
				baseGestureData.right_wrist_positions[i][1],
				baseGestureData.right_wrist_positions[i][2]
			);
			Vector3 normalizedRightPos = rightPos - referencePos;
			baseGestureData.right_wrist_positions[i][0] = normalizedRightPos.x;
			baseGestureData.right_wrist_positions[i][1] = normalizedRightPos.y;
			baseGestureData.right_wrist_positions[i][2] = normalizedRightPos.z;

		}
	}


	public void StartVisualizationVideo(bool continualLoop = false)
	{
		if (isPlaying || baseGestureData == null) return;

		isPlaying = true;
		currentFrame = 0;
		playbackCoroutine = StartCoroutine(PlayGestureVideo(continualLoop));
	}

	public void StopVisualizationVideo()
	{
		if (!isPlaying) return;

		isPlaying = false;
		if (playbackCoroutine != null)
			StopCoroutine(playbackCoroutine);
	}

	private IEnumerator PlayGestureVideo(bool continualLoop)
	{
		int frameCount = baseGestureData.left_joint_rotations.Count;

		bool firstTime = true;

		while (continualLoop || firstTime)
		{
			currentFrame = 0;

			while (isPlaying && currentFrame < frameCount)
			{
				UpdateVisualizationFrame(currentFrame);

				yield return new WaitForSeconds(timeBetweenFrames);

				currentFrame++;
			}

			firstTime = false;

			if (continualLoop)
				yield return new WaitForSeconds(1.0f);
		}

		isPlaying = false;
	}

	public void ResetVideo()
	{
		currentFrame = 0;
		UpdateVisualizationFrame(currentFrame);
	}

	private void UpdateVisualizationFrame(int frameIndex)
	{
		if (baseGestureData == null)
			return;

		Debug.Log($"Showing frame {frameIndex} / {baseGestureData.left_joint_rotations.Count}");

		List<HandJointId> jointIds = JointDataGather.ImportantHandJointIDs();

		for (int i = 0; i < jointIds.Count; i++)
		{
			HandJointId jointId = jointIds[i];

			List<float> leftRotationArray = baseGestureData.left_joint_rotations[frameIndex][i];
			List<float> rightRotationArray = baseGestureData.right_joint_rotations[frameIndex][i];

			Quaternion leftRotation = new Quaternion(leftRotationArray[0], leftRotationArray[1], leftRotationArray[2], leftRotationArray[3]);
			Quaternion rightRotation = new Quaternion(rightRotationArray[0], rightRotationArray[1], rightRotationArray[2], rightRotationArray[3]);

			// Update left hand joint
			if (leftHandVisual != null)
			{
				Transform leftJointTransform = leftHandVisual.GetTransformByHandJointId(jointId);
				if (leftJointTransform != null)
					leftJointTransform.localRotation = leftRotation;
			}

			// Update right hand joint
			if (rightHandVisual != null)
			{
				Transform rightJointTransform = rightHandVisual.GetTransformByHandJointId(jointId);
				if (rightJointTransform != null)
					rightJointTransform.localRotation = rightRotation;
			}
		}

		// Update wrist/root transforms
		if (leftHandVisual != null)
		{
			List<float> leftWristPositionArray = baseGestureData.left_wrist_positions[frameIndex];
			List<float> leftWristRotationArray = baseGestureData.left_wrist_rotations[frameIndex];

			Vector3 leftWristPosition = new Vector3(leftWristPositionArray[0], leftWristPositionArray[1], leftWristPositionArray[2]);
			Quaternion leftWristRotation = new Quaternion(leftWristRotationArray[0], leftWristRotationArray[1], leftWristRotationArray[2], leftWristRotationArray[3]);

			leftHandVisual.transform.localPosition = leftWristPosition;
			leftHandVisual.transform.localRotation = leftWristRotation;
		}

		if (rightHandVisual != null)
		{
			List<float> rightWristPositionArray = baseGestureData.right_wrist_positions[frameIndex];
			List<float> rightWristRotationArray = baseGestureData.right_wrist_rotations[frameIndex];

			Vector3 rightWristPosition = new Vector3(rightWristPositionArray[0], rightWristPositionArray[1], rightWristPositionArray[2]);
			Quaternion rightWristRotation = new Quaternion(rightWristRotationArray[0], rightWristRotationArray[1], rightWristRotationArray[2], rightWristRotationArray[3]);

			rightHandVisual.transform.localPosition = rightWristPosition;
			rightHandVisual.transform.localRotation = rightWristRotation;
		}
	}


}
