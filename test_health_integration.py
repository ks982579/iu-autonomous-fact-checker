#!/usr/bin/env python3
"""
Test script for health check integration between API and Chrome extension
"""
import json
import requests
import time
from pathlib import Path


def test_config_loading():
    """Test that config.json can be loaded"""
    config_path = Path(__file__).parent / "config.json"
    
    print("🔧 Testing config file loading...")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Config loaded successfully")
        print(f"   • API host: {config['api']['host']}")
        print(f"   • API port: {config['api']['port']}")
        print(f"   • Extension poll interval: {config['extension']['health_check_poll_interval_ms']}ms")
        
        # Check extension config exists
        ext_config_path = Path(__file__).parent / "extensions/my-chrome-ext/dist/config.json"
        if ext_config_path.exists():
            print(f"✅ Extension config.json exists in dist folder")
        else:
            print(f"❌ Extension config.json missing from dist folder")
            
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None


def test_api_health_endpoint(config):
    """Test the API health endpoint"""
    if not config:
        return False
        
    base_url = config['api']['base_url']
    timeout = config['extension']['request_timeout_ms'] / 1000  # Convert to seconds
    
    print(f"\n❤️  Testing API health endpoint at {base_url}/health...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint responding")
            print(f"   • Status: {data.get('status', 'unknown')}")
            print(f"   • Timestamp: {data.get('timestamp', 'unknown')}")
            print(f"   • Response time: {response.elapsed.total_seconds():.3f}s")
            return True
        else:
            print(f"❌ Health endpoint returned {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API - is it running?")
        print(f"   Start with: python run_api.py")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Health check timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def simulate_extension_polling(config, duration=10):
    """Simulate extension health check polling"""
    if not config:
        return
        
    base_url = config['api']['base_url']
    interval = config['extension']['health_check_poll_interval_ms'] / 1000
    timeout = config['extension']['request_timeout_ms'] / 1000
    
    print(f"\n🔄 Simulating extension health polling for {duration}s...")
    print(f"   • Polling every {interval}s")
    print(f"   • Request timeout: {timeout}s")
    
    start_time = time.time()
    poll_count = 0
    success_count = 0
    
    while time.time() - start_time < duration:
        poll_count += 1
        
        try:
            response = requests.get(f"{base_url}/health", timeout=timeout)
            if response.status_code == 200:
                success_count += 1
                status = "✅ HEALTHY"
            else:
                status = "❌ UNHEALTHY"
                
            print(f"   Poll #{poll_count}: {status} ({response.elapsed.total_seconds():.3f}s)")
            
        except requests.exceptions.ConnectionError:
            print(f"   Poll #{poll_count}: ❌ CONNECTION ERROR")
        except requests.exceptions.Timeout:
            print(f"   Poll #{poll_count}: ❌ TIMEOUT")
        except Exception as e:
            print(f"   Poll #{poll_count}: ❌ ERROR - {e}")
        
        time.sleep(interval)
    
    success_rate = (success_count / poll_count) * 100 if poll_count > 0 else 0
    print(f"\n📊 Polling Results:")
    print(f"   • Total polls: {poll_count}")
    print(f"   • Successful: {success_count}")
    print(f"   • Success rate: {success_rate:.1f}%")


def main():
    """Run the health integration test"""
    print("🧪 Health Check Integration Test")
    print("=" * 50)
    
    # Test 1: Config loading
    config = test_config_loading()
    
    # Test 2: API health endpoint
    api_healthy = test_api_health_endpoint(config)
    
    # Test 3: Simulate extension polling (only if API is healthy)
    if api_healthy:
        simulate_extension_polling(config, duration=10)
    else:
        print("\n⚠️  Skipping polling simulation - API not healthy")
    
    print("\n🎯 Integration Test Summary:")
    print(f"   • Config loading: {'✅ PASS' if config else '❌ FAIL'}")
    print(f"   • API health check: {'✅ PASS' if api_healthy else '❌ FAIL'}")
    print(f"   • Ready for extension: {'✅ YES' if config and api_healthy else '❌ NO'}")
    
    if config and api_healthy:
        print(f"\n🚀 Next Steps:")
        print(f"   1. Load extension in Chrome from: extensions/my-chrome-ext/dist/")
        print(f"   2. Click extension icon → YES")
        print(f"   3. Watch status bar update every {config['extension']['health_check_poll_interval_ms']}ms")


if __name__ == "__main__":
    main()