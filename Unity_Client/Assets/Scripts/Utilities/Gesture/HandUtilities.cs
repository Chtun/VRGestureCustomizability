using System.Collections.Generic;
using UnityEngine;

public class HandUtilities
{

	public static List<List<List<float>>> ComputeParentRelativeRotations(
		List<List<List<float>>> jointRotations
	)
	{
		int numFrames = jointRotations.Count;
		int numJoints = jointRotations[0].Count;

		// Build output list
		var relativeRotations = new List<List<List<float>>>(numFrames);

		// Map of current joint -> parent joint index
		var connectedIndices = GetConnectedIndicesList();

		for (int f = 0; f < numFrames; f++)
		{
			var frameRotations = new List<List<float>>(numJoints);

			for (int j = 0; j < numJoints; j++)
			{
				var qList = jointRotations[f][j]; // [x,y,z,w]
				Quaternion q = new Quaternion(qList[0], qList[1], qList[2], qList[3]);

				int? parentIdx = connectedIndices[j];
				Quaternion qRelative;

				if (parentIdx == null)
				{
					qRelative = q;
				}
				else
				{
					var parentList = jointRotations[f][parentIdx.Value];
					Quaternion parentQ = new Quaternion(parentList[0], parentList[1], parentList[2], parentList[3]);

					Quaternion parentQInv = Quaternion.Inverse(parentQ);

					// q_relative = q_child * inverse(parent)
					qRelative = q * parentQInv;
				}

				// Convert back to list<float>
				frameRotations.Add(new List<float> { qRelative.x, qRelative.y, qRelative.z, qRelative.w });
			}

			relativeRotations.Add(frameRotations);
		}

		return relativeRotations;
	}


	private static Dictionary<int, int?> GetConnectedIndicesList()
	{
		return new Dictionary<int, int?>()
		{
			{0, 1}, {1, 2}, {2, 3}, {3, null},
			{4, 5}, {5, 6}, {6, 7}, {7, 8}, {8, null},
			{9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, null},
			{14, 15}, {15, 16}, {16, 17}, {17, 18}, {18, null},
			{19, 20}, {20, 21}, {21, 22}, {22, 23}, {23, null}
		};
	}

}
