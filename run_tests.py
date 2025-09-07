# ./run_tests.py
"""
Test runner script for the Fact Checking API
"""
import subprocess
import sys
import os


def run_tests():
    """Run pytest with appropriate settings"""
    
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
        print("Running Fact Checking API Tests")
        print("=" * 50)
        
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("\nAll tests passed!")
        else:
            print(f"\nTests failed with return code {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()