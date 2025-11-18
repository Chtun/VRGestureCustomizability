using UnityEngine;
using UnityEngine.UI;
using TMPro;
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
    public TextMeshProUGUI timerText;
    public ChallengeButton challengeButton;

    private GameObject[] activeTargets = new GameObject[6];  // 2 of each color
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

        if (timerText != null)
            timerText.text = "Time: 0.00s";

        // Spawn 2 red, 2 yellow, 2 blue
        SpawnTarget(redTargetPrefab, 0);
        SpawnTarget(redTargetPrefab, 1);

        SpawnTarget(yellowTargetPrefab, 2);
        SpawnTarget(yellowTargetPrefab, 3);

        SpawnTarget(blueTargetPrefab, 4);
        SpawnTarget(blueTargetPrefab, 5);
    }

    private void SpawnTarget(GameObject prefab, int index)
    {
        if (prefab == null || spawnPoints.Length == 0)
            return;

        Transform spawn = spawnPoints[index % spawnPoints.Length];

        GameObject target = Instantiate(prefab, spawn.position, spawn.rotation);
        activeTargets[index] = target;

        var ft = target.AddComponent<FloatingTarget>();
        ft.Initialize(this, GetExpectedSpellColor(prefab));
    }

    public void TargetDestroyed(GameObject target)
    {
        for (int i = 0; i < activeTargets.Length; i++)
        {
            if (activeTargets[i] == target)
                activeTargets[i] = null;
        }

        // check if all are destroyed
        foreach (var t in activeTargets)
        {
            if (t != null)
                return;
        }

        EndChallenge();
    }

    private void EndChallenge()
    {
        challengeRunning = false;

        if (timerText != null)
            timerText.text = $"Finished! Time: {timer:F2}s";

        if (challengeButton != null)
            challengeButton.ResetButton();

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
