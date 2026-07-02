import onnxruntime as ort
import numpy as np
from time import sleep, time
import sys
import viser

from g1_env import G1Env
from g1_config import ONNX_PATH
from mjlab.viewer.viser.scene import ViserMujocoScene

# Check if ONNX model exists
try:
    with open(ONNX_PATH, "r"): pass
except FileNotFoundError:
    print(f"Error: ONNX model not found at {ONNX_PATH}")
    sys.exit(1)

print("Loading ONNX Model...")
sess = ort.InferenceSession(ONNX_PATH)
input_name = sess.get_inputs()[0].name

print("Initializing Environment (Headless mode for WSL compatibility)...")
# Using render_mode=None avoids GLFW initialization which causes segfaults in WSL
env = G1Env(render_mode=None)
obs, info = env.reset()

print("Starting Viser Server...")
# Start the Viser 3D Web Server on port 8080 (or another available port)
server = viser.ViserServer(port=8080)

# Create the 3D scene visualizer for our MuJoCo model
scene = ViserMujocoScene.create(server=server, mj_model=env.model, num_envs=1)
scene.camera_tracking_enabled = True  # Track the robot's pelvis

print("\n-------------------------------------------------------------")
print(f" Viser 3D Viewer is ready! Please open in your browser:")
print(f" http://localhost:8080")
print("-------------------------------------------------------------\n")

dt = env.dt
try:
    while True:
        start_time = time()
        
        # ONNX expects batch dimension: shape [1, 98]
        obs_batch = obs.reshape(1, -1).astype(np.float32)
        
        # Predict action
        action = sess.run(None, {input_name: obs_batch})[0][0]
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action) 
        
        # Send physical state updates to the Viser browser viewer
        scene.update_from_mjdata(env.data)
        
        # Maintain real-time simulation speed
        elapsed = time() - start_time
        sleep_time = max(0.0, dt - elapsed)
        sleep(sleep_time)
        
        if terminated or truncated:
            print("Episode ended. Resetting environment...\n")
            obs, info = env.reset()
            
except KeyboardInterrupt:
    print("\nShutting down viewer...")
finally:
    server.stop()
    env.close()