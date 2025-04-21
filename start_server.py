import subprocess
import os
import sys
import time

def run_backend():
    """Starts the Flask backend server."""
    print("Starting Flask Backend...")
    
    # Get the absolute path of the backend app.py
    backend_path = os.path.join(os.path.abspath("backend"), "app.py")
    print(f"Backend script path: {backend_path}")  # Print the backend path for debugging
    
    # Ensure the working directory is correctly set to the backend folder
    backend_process = subprocess.Popen(["python", backend_path], cwd=os.path.abspath("backend"))
    return backend_process

def run_frontend():
    """Starts the React frontend server."""
    print("Starting React Frontend...")
    frontend_process = subprocess.Popen(["npm", "start"], cwd=os.path.abspath("frontend"))
    return frontend_process

def main():
    backend_process = run_backend()
    time.sleep(2)  # Wait a bit for the backend to start
    frontend_process = run_frontend()

    try:
        # Keep both servers running
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping the servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
