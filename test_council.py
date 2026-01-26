#!/usr/bin/env python3
"""Comprehensive tests for council implementation."""
import sys
import subprocess
import os

def run_test(name: str, cmd: str) -> bool:
    print(f"\n{'='*50}")
    print(f"TEST: {name}")
    print(f"{'='*50}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        if result.returncode != 0:
            print(f"❌ FAILED (exit code {result.returncode})")
            return False
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    # Change to the council directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("="*50)
    print("COUNCIL TEST SUITE")
    print("="*50)
    print(f"Working dir: {os.getcwd()}")

    # Tests that don't require API calls
    no_cost_tests = [
        ("1. Syntax check", "python -m py_compile crew.py"),
        ("2. Import check", "python -c 'import crew'"),
        ("3. Provider check", "python -c 'from crew import check_providers; print(check_providers())'"),
        ("4. Tool test", "python -c 'from crew import write_file, WORKSPACE_DIR; print(write_file._run(\"test.txt\", \"test\")); (WORKSPACE_DIR / \"test.txt\").unlink()'"),
        ("5. Agent creation", "python -c 'from crew import create_agents; agents = create_agents(); print(f\"Created {len(agents)} agents\"); [print(f\"  - {a.role}\") for a in agents]'"),
        ("6. Crew creation", "python -c 'from crew import create_crew; c = create_crew(\"test\"); print(f\"Agents: {len(c.agents)}, Tasks: {len(c.tasks)}\")'"),
    ]

    print("\n" + "="*50)
    print("PHASE 1: NO-COST TESTS")
    print("="*50)

    passed = 0
    failed = 0
    for name, cmd in no_cost_tests:
        if run_test(name, cmd):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"PHASE 1 RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    if failed > 0:
        print("\n⚠️  Fix the above failures before running API tests!")
        return 1

    print("\n✓ All no-cost tests passed!")
    print("Run with --api flag to test actual API calls")

    # Check for --api flag
    if "--api" in sys.argv:
        print("\n" + "="*50)
        print("PHASE 2: API TESTS (costs money)")
        print("="*50)

        # Single agent test
        api_test = '''python -c "
from crewai import Agent, Task, Crew, Process

agent = Agent(
    role='Tester',
    goal='Say hello',
    backstory='You are a test agent.',
    llm='anthropic/claude-opus-4-5-20251101',
    verbose=False
)

task = Task(
    description='Respond with exactly: TEST_PASSED',
    expected_output='TEST_PASSED',
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
result = crew.kickoff()
print(f'Result: {result}')
assert 'TEST_PASSED' in str(result) or 'test' in str(result).lower(), 'Unexpected result'
print('API test passed!')
"'''
        if run_test("7. Single agent API", api_test):
            passed += 1
        else:
            failed += 1

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
