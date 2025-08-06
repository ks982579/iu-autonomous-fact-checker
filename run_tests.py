#!/usr/bin/env python3
"""
Test runner script for the Fact Checking API
"""
import subprocess
import sys
import os


def run_tests():
    """Run pytest with appropriate settings"""
    
    # Activate virtual environment
    venv_path = os.path.join(os.path.dirname(__file__), '.venv', 'bin', 'activate')
    
    # Command to run tests
    cmd = [
        sys.executable, '-m', 'pytest', 
        'tests/', 
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--color=yes',  # Colorized output
        '--durations=10'  # Show 10 slowest tests
    ]
    
    try:
        print("🧪 Running Fact Checking API Tests")
        print("=" * 50)
        
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()