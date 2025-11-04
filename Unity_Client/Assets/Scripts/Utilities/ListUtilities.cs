using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using System.Text;

public static class ListUtilities
{
	public static void AppendNestedList(StringBuilder sb, List<List<List<float>>> data)
	{
		foreach (var frame in data)
		{
			sb.Append("[ ");
			foreach (var joint in frame)
			{
				sb.Append("[");
				sb.Append(string.Join(", ", joint));
				sb.Append("] ");
			}
			sb.AppendLine("]");
		}
	}

	public static void Append2DList(StringBuilder sb, List<List<float>> data)
	{
		foreach (var frame in data)
		{
			sb.AppendLine("[ " + string.Join(", ", frame) + " ]");
		}
	}

	public static List<List<List<float>>> Parse3DList(JArray array)
	{
		var outerList = new List<List<List<float>>>();

		foreach (var frameToken in array)
		{
			var frameArray = (JArray)frameToken;
			var frameList = new List<List<float>>();

			foreach (var jointToken in frameArray)
			{
				var jointArray = (JArray)jointToken;
				var jointList = new List<float>();

				foreach (var valueToken in jointArray)
					jointList.Add((float)valueToken);

				frameList.Add(jointList);
			}

			outerList.Add(frameList);
		}

		return outerList;
	}

	public static List<List<float>> Parse2DList(JArray array)
	{
		var outerList = new List<List<float>>();

		foreach (var frameToken in array)
		{
			var frameArray = (JArray)frameToken;
			var frameList = new List<float>();

			foreach (var valueToken in frameArray)
				frameList.Add((float)valueToken);

			outerList.Add(frameList);
		}

		return outerList;
	}

}
