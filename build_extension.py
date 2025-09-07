"""
Extension build script - In Python for consistency w/project
Builds the Chrome extension with synchronized config from root directory
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path


def load_config():
    """Load and validate the root config file"""
    config_path = Path(__file__).parent / "config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['api', 'extension']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        return config
    
    except FileNotFoundError:
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config.json: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def sync_config(_config):
    """Copy config.json to extension's public directory"""
    root_config = Path(__file__).parent / "config.json"
    ext_config = Path(__file__).parent / "extensions/my-chrome-ext/public/config.json"
    
    try:
        # Create directory if it doesn't exist
        ext_config.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy config file
        shutil.copy2(root_config, ext_config)
        
        print(f"Config synced: {root_config} → {ext_config}")
        return True
    
    except Exception as e:
        print(f"Error syncing config: {e}")
        return False


def build_extension():
    """Build the Chrome extension using bun"""
    ext_dir = Path(__file__).parent / "extensions/my-chrome-ext"
    
    if not ext_dir.exists():
        print(f"Error: Extension directory not found: {ext_dir}")
        return False
    
    try:
        print(f"Building extension in {ext_dir}...")
        
        # Change to extension directory and run build command
        result = subprocess.run(
            ["bun", "run", "build:extension"],
            cwd=ext_dir,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode == 0:
            print("Extension build completed successfully!")
            
            # Show build output if verbose
            if result.stdout.strip():
                print("Build output:")
                for line in result.stdout.strip().split('\n'):
                    print(f"   {line}")
            
            return True
        else:
            print("Extension build failed!")
            if result.stderr.strip():
                print("Error output:")
                for line in result.stderr.strip().split('\n'):
                    print(f"   {line}")
            return False
    
    except subprocess.TimeoutExpired:
        # I haven't run into error with build over 60sec.
        print("Error: Build timed out after 60 seconds")
        return False
    except FileNotFoundError:
        print("Error: 'bun' command not found. Make sure bun is installed and in PATH")
        print("You can update the commands to run with npm if you would like - requires more effort.")
        return False
    except Exception as e:
        print(f"Error during build: {e}")
        return False


def verify_build():
    """Verify the build output"""
    dist_dir = Path(__file__).parent / "extensions/my-chrome-ext/dist"
    
    ## NOTE: updating extension may require updates to verification!
    required_files = [
        "manifest.json",
        "popup.html", 
        "popup.js",
        "content.js",
        "config.json"
    ]
    
    print("Verifying build output...")
    
    if not dist_dir.exists():
        print(f"Error: Dist directory not found: {dist_dir}")
        return False
    
    missing_files = []
    for file_name in required_files:
        file_path = dist_dir / file_name
        if file_path.exists():
            print(f"   {file_name}")
        else:
            print(f"   {file_name} (missing)")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"Build verification failed. Missing files: {', '.join(missing_files)}")
        return False
    
    print("Build verification passed!")
    return True


def show_next_steps(config):
    """Show next steps for loading the extension"""
    dist_path = Path(__file__).parent / "extensions/my-chrome-ext/dist"
    api_url = config['api']['base_url']
    
    print("\nNext Steps:")
    print("=" * 50)
    print("1. Load extension in Chrome:")
    print(f"   - Go to chrome://extensions/")
    print(f"   - Enable 'Developer mode'")
    print(f"   - Click 'Load unpacked'")
    print(f"   - Select: {dist_path.absolute()}")
    print()
    print("2. Start the API server:")
    print(f"   - Run: python run_api.py")
    print(f"   - API will start at: {api_url}")
    print()
    print("3. Test the integration:")
    print("   - Click extension icon in Chrome toolbar")
    print("   - Click 'YES' to open fact-checker modal")
    print(f"   - Status bar should update every {config['extension']['health_check_poll_interval_ms']}ms")
    print("NOTE: It is easier to use if you 'pin' the extension in Chrome (only supported browser currently)")


def main():
    """Main build process"""
    print("Chrome Extension Build Script")
    print("=" * 50)
    
    # Step 1: Load and validate config
    print("Loading configuration...")
    config = load_config()
    print(f"  Config loaded (API: {config['api']['base_url']})")
    
    # Step 2: Sync config to extension
    print("\nSyncing configuration...")
    if not sync_config(config):
        sys.exit(1)
    
    # Step 3: Build extension
    print("\nBuilding Chrome extension...")
    if not build_extension():
        sys.exit(1)
    
    # Step 4: Verify build
    print("\nVerifying build...")
    if not verify_build():
        sys.exit(1)
    
    # Step 5: Show next steps
    show_next_steps(config)
    
    print("\nExtension build completed successfully!")


if __name__ == "__main__":
    main()