# Custom Recognition System for Two-Hand, Dynamic, Motile Gestures

This builds upon the implementation of Gesture Builder by additionally including two-hand gesture recognition, in which the posture and movement of the hands are both factors of custom gestures. This handles hand postures over time (dynamic) and hand motion (motile) for both hands. Single-hand gestures are not currently supported.

## Features

**Custom Gesture Recognition System:**
- Communication between Meta Quest headset and edge server to stream hand positions and receive gesture match scores in real time (every 0.1s).
- Saving custom gesture data to server.
- Retrieving stored data for default and custom gestures.
- Removing selected stored custom gestures.

**Game Task for Testing:**
- UI for recording custom gestures.
- UI for practicing custom or default gestures and displaying gesture match scores.
- Replay mode for visualizing each custom or default gesture.
- UI for removing custom gestures.
- Game mode where player may cast spells or take actions, where each gesture type is bound to a particular spell/action.
- Game task which provides the player an experience to test their skills with the actions/spells.
- Debugging logs which includes logs over what gestures were recognized and task metrics like time to completion and target hits

## Installation

### Requirements

- Meta Quest headset (Tested on Meta Quest 3).
- Desktop PC/Server/Edge device to run gesture recognition server.
- Unity Editor Version 6000.2.7f2 or higher (Tested on 6000.2.7f2).
- Python 3.12 or higher.
- Packages in requirements.txt (or alternatively, setup.py) such as pytorch, scikit-learn, numpy, uvicorn, etc.
- Tested on Windows PC, may not work for Mac/Linux due to Unity-Meta Quest compatibility issues.

### Installation Steps  

1. Clone the repository:
```bash
git clone https://github.com/Chtun/VRGestureCustomizability.git
cd VRGestureCustomizability
```

2. Install dependencies:
```bash
cd GestureRecognition
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

4. Install Unity:
    - Download and install **Unity Hub** from the [Official Unity Download Page](https://unity.com).
    - Follow the step-by-step instructions in the [Unity Installation Guide](https://unity3d.com) to install your preferred editor version.
    - Ensure you include the **Linux/Mac/Windows Build Support** modules depending on your target OS.
    - Recommend installing version **6000.2.7f2** as project is stable in this version.

5. Add the Unity Client project to Unity Hub:
   - Open **Unity Hub**.
   - Click the **Add** button (or **Open** -> **Add project from disk**).
   - Navigate to the root directory of this repository.
   - Select the `Unity_Client` folder and click **Add Project**.
   - Open the project using the Unity Editor version recommended in the guide above.

6. Connect Meta Quest to Unity Editor:
   - Enable Developer Mode on your headset using the [Meta Quest Developer Guide](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/).
   - Install and configure the Meta Horizon Link desktop app on your PC by following the [Meta Quest Link Setup](https://developers.meta.com/horizon/documentation/unity/unity-link/).
   - Configure the Unity project settings for OpenXR testing by following the official [Unity Manual for Meta Quest Link](https://docs.unity3d.com/Packages/com.unity.xr.meta-openxr@2.1/manual/get-started/link.html).
   

## Getting Started

Here's an example of how to run the gesture recognition server and game client on the Meta Quest 3 Headset.

### Gesture Recognition Server

1. **Edit the config.yaml file (optional):**
- `server_settings\host` is the host name for the server.
- `server_settings\port` is the port number for the server.
- `paths\data_folder` contains root directory for hand joint sequence data for each default gesture.
- `paths\output_folder` contains the output folder for hand animations and root directory for the VQ-VAE hand posture model weights.
- `paths\input_VQVAE_model` contains the name and extension of the VQ-VAE hand posture model weights file.
- `paths\gesture_template_json` contains the path to the custom gesture template json file.
- `gesture_template_paths` contains pairs of spell/action name and hand joint sequence data path for each default gesture.
- `gesture_settings\BUFFER_MAX_LEN` is the maximum buffer length for the stream of incoming hand joint data. For an incoming stream rate of 0.1s and BUFFER_MAX_LEN = 30, this means that the most recent 3 seconds of data is stored.
- `gesture_settings\MATCH_THRESHOLD` is the match distance score to be below for the stream of incoming hand joint data to match with a gesture that it is compared to. For example, if the distance score between the stream of incoming hand joint data and a particular gesture is 1.0 and the MATCH_THRESHOLD is 1.6, then the stream of incoming hand joint data is labeled with that particular gesture's name.
- Do not alter `model_params` unless the VQ-VAE hand posture model architecture is altered.

2. **Run the gesture recognition server:**

```bash
cd server
python gesture_server.py
```

### Game Client

1. **Connect Meta Quest to Desktop PC**:
    - Connect the Meta Quest headset to Desktop PC via wired connection.
    - Ensure Meta Quest Link is enabled and Meta Quest headset is in developer environment.

2. **Start Unity_Client Project:** Open Unity Hub, then open the Unity_Client project.


3. **Edit Configs If Necessary:**
    - Under Unity Project Folder Directory panel, find the `config.json` file under `Assets\StreamingAssets` folder.
    - The config file contains:
        - host and port number of the server.
        - Participant's name or ID to appropriately name the debug logs.
        - manualOverride allows for the keyboard to be used to cast spells if set to true.
        - defaultGestures contains the mapping of spell/action display name to label ID.

4. **Run Unity_Client Project:** 
    - Press the Play button in the Unity Editor environment.
    - This should start up the client in the Meta Quest headset.

5. **Explore the Game Client:**
    - On start, the main menu will open up.
    - Default Gesture and Custom Gesture buttons toggle which mode is used.
    - A button for recording gestures is available, which takes users to a new scene where users can record custom gestures for each cast spell/action type.
    - A button for practicing gestures is available, which takes users to a new scene where users can practice their gestures and see what match score is given for each gesture when compared to their recent stream of hand joint data. In the practice mode, the first stored hand joint sequence data of each gesture type may also be visualized.
    - A button for starting the game task. In the game task, players can actually cast the spells.
        - The Show Teleport spell toggles the teleport on and off; if the teleport indicator is off and the spell is cast, then the teleport indicator turns on, and if the teleport indicator is on and the spell is cast, then the teleport indicator turns off. The teleport indicator looks like a ring of purple orbs/cubes that are floating and will show exactly where on the floor you will teleport to if Cast Teleport is used.
        - All other spells cast when used, except for Cast Teleport which can only cast if the teleport indicator is shown. Spells cast immediately when a gesture match is found.

6. **Complete Game Task:**
    - First, the player must navigate to the game task environment.
    - Then, to complete the game task, the player must teleport and/or move towards the small table with a button on it with a glowing effect around it, with text above it saying that the button must be pressed to start the game task.
    - When the game task starts, the clock immediately starts, so the player must be ready. Once the button is hit, several hovering red, yellow, and blue cylinders will spawn. If the player is standing where they can read the text when they hit the button, the position of the cylinders will all be directly behind the player, so the player must turn around 180 degrees to view the different cylinders.
    - There should be roughly 2-3 different cylinders per color. The red cylinders must be hit by the fireball from the Cast Fireball spell, the yellow cylinders must be hit by the lightning from the Cast Lightning spell, and the blue cylinders must be hit by the iceball from the Cast Ice spell. Each cylinder will clear away once it has been hit by the appropriate spell.
    - Once all cylinders have been cleared, the task is finished. Players should try to finish as quick as possible to reduce their time-to-completion.
    - The debug logs contains when the game task begins, the different spells cast, when each cylinder is hit, and when the task is completed.
    

### Debug logs

The debug logs can be found under the folder defined by the common Unity data path (`Application.persistentDataPath`), which is usually found at: `C:\Users\<YourUsername>\AppData\LocalLow\<CompanyName>\<ProductName>\`, or for this project, `C:\Users\<YourUsername>\AppData\LocalLow\DefaultCompany\Unity_VR_Template`, where `<Your Username>` is your Windows account username.

## Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{VRGestureCustomizability,
  title = {Custom Gesture Recognition System},
  author = {Chitsein Htun},
  year = {2026},
  url = {https://github.com/Chtun/VRGestureCustomizability}
}
```

## Contact

For questions and support:
- Create an issue on GitHub
- Email: chtun@live.com

## Acknowledgments

This project is based on research by Chitsein Htun and builds upon the concepts presented in "XXX".

