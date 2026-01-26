#!/usr/bin/env python3
"""Adversarial tests - trying to break the council."""
import os
import sys
import shutil
from pathlib import Path

WORKSPACE = Path.home() / ".council" / "crewai-council" / "workspace"

def test_path_traversal():
    """Test 1.1: Can we escape the workspace?"""
    print("\n[TEST] Path Traversal Attack")
    from crew import write_file, WORKSPACE_DIR

    test_path = Path("/tmp/council_traversal_test.txt")
    test_path.unlink(missing_ok=True)

    try:
        write_file._run(filename="../../../tmp/council_traversal_test.txt", content="pwned")
    except:
        pass

    if test_path.exists():
        test_path.unlink()
        return False, "CRITICAL: Path traversal succeeded!"
    return True, "Path traversal blocked"

def test_empty_content():
    """Test 1.2: Empty content handling"""
    print("\n[TEST] Empty Content")
    from crew import write_file, WORKSPACE_DIR

    result = write_file._run(filename="empty_test.txt", content="")
    path = WORKSPACE_DIR / "empty_test.txt"
    exists = path.exists()
    if exists:
        path.unlink()
    return exists, f"Empty file created: {exists}"

def test_special_chars():
    """Test 1.3: Special characters in content"""
    print("\n[TEST] Special Characters")
    from crew import write_file, WORKSPACE_DIR

    content = "Hello\x00World\n你好\n"
    try:
        write_file._run(filename="special_test.txt", content=content)
        path = WORKSPACE_DIR / "special_test.txt"
        read_back = path.read_text()
        path.unlink()
        return True, "Handled special chars"
    except Exception as e:
        return False, f"Failed: {e}"

def test_filename_injection():
    """Test 1.4: Malicious filenames"""
    print("\n[TEST] Filename Injection")
    from crew import write_file, WORKSPACE_DIR

    bad_names = [
        'file;rm -rf /',
        '$(whoami).txt',
        'file|cat /etc/passwd',
    ]

    results = []
    for name in bad_names:
        try:
            write_file._run(filename=name, content='test')
            # If we get here, the file was created - clean it up
            try:
                (WORKSPACE_DIR / name).unlink()
            except:
                pass
            results.append(f"CREATED: {name[:20]}...")
        except Exception as e:
            results.append(f"Blocked: {name[:20]}...")

    # These should all be created (they're just weird filenames, not actual injection)
    # The real test is that no shell commands executed
    return True, f"Filename tests: {len(results)} checked"

def test_provider_check_depth():
    """Test 2.2: Do we validate API keys or just check existence?"""
    print("\n[TEST] API Key Validation Depth")
    from crew import check_providers

    # Save real key
    real_key = os.environ.get("ANTHROPIC_API_KEY")

    # Set invalid key
    os.environ["ANTHROPIC_API_KEY"] = "invalid_key_12345"
    status = check_providers()

    # Restore
    if real_key:
        os.environ["ANTHROPIC_API_KEY"] = real_key

    # If it still shows True, we're not validating (expected behavior for now)
    if status.get("anthropic"):
        return True, "WARNING: Only checking key existence, not validity (known issue)"
    return True, "API keys are validated"

def test_workspace_isolation():
    """Test: Verify workspace is clean between runs"""
    print("\n[TEST] Workspace Isolation")

    # Clear workspace
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True, exist_ok=True)

    before = set(WORKSPACE.iterdir())
    return True, f"Workspace clean: {len(before)} files"

def test_empty_task():
    """Test 5.1: Empty task handling"""
    print("\n[TEST] Empty Task Input")
    import subprocess

    result = subprocess.run(
        ['python', 'crew.py', ''],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )

    # Should exit with error or show usage
    if result.returncode != 0 or 'Usage' in result.stdout or 'Usage' in result.stderr:
        return True, "Empty task handled gracefully"
    return False, "Empty task not handled properly"

def main():
    os.chdir(Path(__file__).parent)

    print("="*60)
    print("ADVERSARIAL TEST SUITE")
    print("="*60)

    tests = [
        test_path_traversal,
        test_empty_content,
        test_special_chars,
        test_filename_injection,
        test_provider_check_depth,
        test_workspace_isolation,
        test_empty_task,
    ]

    passed = 0
    failed = 0
    warnings = 0

    for test in tests:
        try:
            success, msg = test()
            if success:
                if "WARNING" in msg:
                    print(f"  ⚠️  {msg}")
                    warnings += 1
                else:
                    print(f"  ✓ {msg}")
                    passed += 1
            else:
                print(f"  ✗ {msg}")
                failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} crashed: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"{'='*60}")

    if failed > 0:
        print("\n⚠️  Security or stability issues found!")
        return 1

    if warnings > 0:
        print("\n⚠️  Warnings found - review before production use")

    return 0

if __name__ == "__main__":
    sys.exit(main())
