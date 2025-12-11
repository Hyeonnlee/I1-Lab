"""
Main execution script for running all agents in sequence
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_name):
    """
    Execute a Python script and handle errors
    
    Args:
        script_name: Name of the script to execute
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        print(f"\n✓ {script_name} completed successfully\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_name}")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"\n✗ File not found: {script_name}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        return False


def main():
    """
    Execute all agents in the specified order
    """
    print("\n" + "="*60)
    print("STARTING MULTI-AGENT PIPELINE")
    print("="*60)
    
    # Define execution order
    scripts = [
        "1_normalization_agent.py",
        "2_query_agent.py",
        "create_csv_data.py",
        "3_visualization_agent.py",
        "4_analysis_agent.py",
        "5_report_agent.py",
        "6_action_agent.py"
    ]
    
    # Track execution status
    success_count = 0
    failed_scripts = []
    
    # Execute each script in order
    for script in scripts:
        if run_script(script):
            success_count += 1
        else:
            failed_scripts.append(script)
            # Decide whether to continue or stop on error
            print(f"\nWarning: {script} failed. Continuing with next script...")
            # Uncomment the line below to stop on first error:
            # break
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Total scripts: {len(scripts)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print("\nFailed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
    else:
        print("\n✓ All scripts completed successfully!")
    
    print("="*60 + "\n")
    
    return len(failed_scripts) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)