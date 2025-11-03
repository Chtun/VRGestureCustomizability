using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class TargetChallengeManager : MonoBehaviour
{
    [Header("Target Prefabs")]
    public GameObject redTargetPrefab;
    public GameObject yellowTargetPrefab;
    public GameObject blueTargetPrefab;

    [Header("Target Spawn Points")]
    public Transform[] spawnPoints;

    [Header("UI Elements")]
    public Text timerText;

    private GameObject[] activeTargets = new GameObject[3];
    private float timer = 0f;
    private bool challengeRunning = false;

    void Update()
    {
        if (challengeRunning)
        {
            timer += Time.deltaTime;
            if (timerText != null)
                timerText.text = $"Time: {timer:F2}s";
        }
    }

    public void StartChallenge()
    {
        if (challengeRunning) return;

        Debug.Log("Challenge started!");
        timer = 0f;
        challengeRunning = true;

        // Spawn 3 targets at random or predefined spawn points
        SpawnTarget(redTargetPrefab, 0);
        SpawnTarget(yellowTargetPrefab, 1);
        SpawnTarget(blueTargetPrefab, 2);
    }

    private void SpawnTarget(GameObject prefab, int index)
    {
        if (prefab == null || spawnPoints.Length == 0) return;

        Transform spawn = spawnPoints[index % spawnPoints.Length];
        GameObject target = Instantiate(prefab, spawn.position, spawn.rotation);
        activeTargets[index] = target;

        var ft = target.AddComponent<FloatingTarget>();
        ft.Initialize(this, GetExpectedSpellColor(prefab));
    }

    public void TargetDestroyed(GameObject target)
    {
        Debug.Log($"Target {target.name} destroyed!");
        for (int i = 0; i < activeTargets.Length; i++)
        {
            if (activeTargets[i] == target)
                activeTargets[i] = null;
        }

        // Check if all targets are gone
        bool allDestroyed = true;
        foreach (var t in activeTargets)
        {
            if (t != null)
            {
                allDestroyed = false;
                break;
            }
        }

        if (allDestroyed)
            EndChallenge();
    }

    private void EndChallenge()
    {
        challengeRunning = false;
        Debug.Log($"Challenge complete! Final time: {timer:F2}s");
    }

    private string GetExpectedSpellColor(GameObject prefab)
    {
        if (prefab == redTargetPrefab) return "Fire";
        if (prefab == yellowTargetPrefab) return "Lightning";
        if (prefab == blueTargetPrefab) return "Ice";
        return "";
    }
}
