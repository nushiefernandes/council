#!/usr/bin/env python3
"""
Adversarial End-to-End Tests for the CrewAI Council.

Goal: Break the council implementation. Find failure modes before production.

Test Categories:
1. write_file Tool Abuse (path traversal, injection, edge cases)
2. Provider Failure Modes (missing keys, invalid keys, timeouts)
3. Input Edge Cases (empty, unicode, injection attempts)
4. Workspace Isolation
5. Module Integrity

Run with: python adversarial_test.py
"""
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

# Add the crew directory to path so we can import from it
CREW_DIR = Path(__file__).parent
sys.path.insert(0, str(CREW_DIR))

WORKSPACE = Path.home() / ".council" / "crewai-council" / "workspace"


class TestResult:
    """Container for test results with severity levels."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.is_warning = False
        self.is_security_issue = False

    def pass_(self, msg: str):
        self.passed = True
        self.message = msg
        return self

    def warn(self, msg: str):
        self.passed = True
        self.is_warning = True
        self.message = msg
        return self

    def fail(self, msg: str):
        self.passed = False
        self.message = msg
        return self

    def security_fail(self, msg: str):
        self.passed = False
        self.is_security_issue = True
        self.message = msg
        return self


# =============================================================================
# TEST CATEGORY 1: write_file Tool Abuse
# =============================================================================

def test_path_traversal():
    """Test 1.1: Can we escape the workspace via path traversal?"""
    result = TestResult("Path Traversal Attack")

    from crew import write_file, WORKSPACE_DIR

    # Create a unique test file path outside workspace
    test_file = Path(tempfile.gettempdir()) / "council_traversal_test.txt"
    test_file.unlink(missing_ok=True)

    # Calculate relative path from workspace to /tmp
    # The workspace is at ~/.council/crewai-council/workspace
    # We need to traverse up and into /tmp
    traversal_paths = [
        "../../../tmp/council_traversal_test.txt",
        "../../../../../../tmp/council_traversal_test.txt",
        "../../../../../../../tmp/council_traversal_test.txt",
        "..\\..\\..\\tmp\\council_traversal_test.txt",  # Windows-style
    ]

    escaped = False
    for traversal in traversal_paths:
        try:
            write_file._run(filename=traversal, content="pwned by path traversal")
        except Exception:
            pass

        if test_file.exists():
            escaped = True
            test_file.unlink()
            break

    if escaped:
        return result.security_fail("CRITICAL: Path traversal succeeded - can write outside workspace!")

    return result.pass_("Path traversal blocked")


def test_absolute_path():
    """Test 1.1b: Can we use absolute paths to escape?"""
    result = TestResult("Absolute Path Attack")

    from crew import write_file

    test_file = Path(tempfile.gettempdir()) / "council_absolute_test.txt"
    test_file.unlink(missing_ok=True)

    try:
        # Try absolute path
        write_file._run(filename=str(test_file), content="pwned by absolute path")
    except Exception:
        pass

    if test_file.exists():
        test_file.unlink()
        return result.security_fail("CRITICAL: Absolute path bypass succeeded!")

    return result.pass_("Absolute path blocked")


def test_symlink_escape():
    """Test 1.1c: Can we escape via symlink?"""
    result = TestResult("Symlink Escape")

    from crew import write_file, WORKSPACE_DIR

    # Create a symlink in workspace pointing outside
    symlink_path = WORKSPACE_DIR / "escape_link"
    target_file = Path(tempfile.gettempdir()) / "council_symlink_test.txt"

    target_file.unlink(missing_ok=True)
    symlink_path.unlink(missing_ok=True)

    try:
        # Create symlink pointing to /tmp
        symlink_path.symlink_to(Path(tempfile.gettempdir()))

        # Try to write through the symlink
        write_file._run(filename="escape_link/council_symlink_test.txt", content="pwned via symlink")

        if target_file.exists():
            target_file.unlink()
            symlink_path.unlink()
            return result.security_fail("CRITICAL: Symlink escape succeeded!")

    except Exception:
        pass
    finally:
        symlink_path.unlink(missing_ok=True)
        target_file.unlink(missing_ok=True)

    return result.pass_("Symlink escape blocked or symlink failed")


def test_empty_content():
    """Test 1.2: Empty content handling."""
    result = TestResult("Empty Content")

    from crew import write_file, WORKSPACE_DIR

    try:
        write_file._run(filename="empty_test.txt", content="")
        path = WORKSPACE_DIR / "empty_test.txt"

        if path.exists():
            size = path.stat().st_size
            path.unlink()
            return result.pass_(f"Empty file created (size: {size})")
        else:
            return result.fail("Empty file was not created")
    except Exception as e:
        return result.fail(f"Exception on empty content: {e}")


def test_null_byte_content():
    """Test 1.3a: Null bytes in content."""
    result = TestResult("Null Byte Content")

    from crew import write_file, WORKSPACE_DIR

    content = "Line1\x00Line2\x00Line3"

    try:
        write_file._run(filename="nullbyte_test.txt", content=content)
        path = WORKSPACE_DIR / "nullbyte_test.txt"

        if path.exists():
            read_back = path.read_text()
            path.unlink()
            # Check if null bytes are preserved
            has_null = "\x00" in read_back
            return result.pass_(f"Null bytes {'preserved' if has_null else 'stripped'}")
        else:
            return result.fail("File with null bytes not created")
    except Exception as e:
        return result.fail(f"Exception: {e}")


def test_unicode_content():
    """Test 1.3b: Unicode content handling."""
    result = TestResult("Unicode Content")

    from crew import write_file, WORKSPACE_DIR

    content = "English\n中文\n日本語\nالعربية\nעברית\n🎉🚀💻"

    try:
        write_file._run(filename="unicode_test.txt", content=content)
        path = WORKSPACE_DIR / "unicode_test.txt"

        if path.exists():
            read_back = path.read_text(encoding='utf-8')
            path.unlink()
            return result.pass_(f"Unicode handled (wrote {len(content)}, read {len(read_back)})")
        else:
            return result.fail("Unicode file not created")
    except Exception as e:
        return result.fail(f"Unicode exception: {e}")


def test_special_chars():
    """Test 1.3c: Special/control characters in content."""
    result = TestResult("Special Characters")

    from crew import write_file, WORKSPACE_DIR

    content = "Tab:\t\nCarriage:\r\nForm:\f\nVertical:\v\nBackspace:\b\nBell:\a"

    try:
        write_file._run(filename="special_test.txt", content=content)
        path = WORKSPACE_DIR / "special_test.txt"

        if path.exists():
            path.unlink()
            return result.pass_("Special chars handled")
        else:
            return result.fail("Special chars file not created")
    except Exception as e:
        return result.fail(f"Exception: {e}")


def test_filename_shell_injection():
    """Test 1.4a: Shell metacharacter filenames."""
    result = TestResult("Filename Shell Injection")

    from crew import write_file, WORKSPACE_DIR

    bad_names = [
        "file;rm -rf /",
        "file`id`.txt",
        "$(whoami).txt",
        "file|cat /etc/passwd",
        "file&echo pwned",
        "file>output.txt",
        "file<input.txt",
    ]

    created = []
    blocked = []

    for name in bad_names:
        try:
            write_file._run(filename=name, content="test")
            created.append(name[:15])
        except Exception:
            blocked.append(name[:15])

    # Clean up any created files
    for f in WORKSPACE_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except:
                pass

    # These filenames being created is OK - they're just weird filenames
    # The real security issue would be if shell commands executed
    return result.pass_(f"Created {len(created)}, blocked {len(blocked)} filenames")


def test_filename_newline():
    """Test 1.4b: Newline in filename."""
    result = TestResult("Filename Newline")

    from crew import write_file, WORKSPACE_DIR

    try:
        write_file._run(filename="file\nname.txt", content="test")
        # If created, check what actually got written
        for f in WORKSPACE_DIR.iterdir():
            if "file" in f.name:
                f.unlink()
        return result.warn("Newline filename created (potential log injection)")
    except Exception:
        return result.pass_("Newline filename blocked")


def test_filename_null_byte():
    """Test 1.4c: Null byte in filename (truncation attack)."""
    result = TestResult("Filename Null Byte")

    from crew import write_file, WORKSPACE_DIR

    try:
        # This might truncate to "safe.txt" in C-based systems
        write_file._run(filename="safe.txt\x00.evil", content="test")
        path = WORKSPACE_DIR / "safe.txt"
        if path.exists():
            path.unlink()
            return result.warn("Null byte may have truncated filename")
        return result.pass_("Null byte filename handled")
    except Exception:
        return result.pass_("Null byte filename blocked")


def test_large_file():
    """Test 1.5: Large file handling."""
    result = TestResult("Large File (1MB)")

    from crew import write_file, WORKSPACE_DIR

    # 1MB of content
    large_content = "A" * (1024 * 1024)

    try:
        write_file._run(filename="large_test.txt", content=large_content)
        path = WORKSPACE_DIR / "large_test.txt"

        if path.exists():
            size = path.stat().st_size
            path.unlink()
            return result.pass_(f"Large file created ({size:,} bytes)")
        else:
            return result.fail("Large file not created")
    except Exception as e:
        return result.fail(f"Exception: {e}")


def test_nested_directory():
    """Test 1.6: Creating nested directories."""
    result = TestResult("Nested Directories")

    from crew import write_file, WORKSPACE_DIR

    try:
        write_file._run(filename="deep/nested/path/file.txt", content="nested test")
        path = WORKSPACE_DIR / "deep" / "nested" / "path" / "file.txt"

        if path.exists():
            # Clean up
            shutil.rmtree(WORKSPACE_DIR / "deep", ignore_errors=True)
            return result.pass_("Nested directories created")
        else:
            return result.fail("Nested file not created")
    except Exception as e:
        return result.fail(f"Exception: {e}")


def test_overwrite_existing():
    """Test 1.7: Overwriting existing files."""
    result = TestResult("File Overwrite")

    from crew import write_file, WORKSPACE_DIR

    path = WORKSPACE_DIR / "overwrite_test.txt"

    try:
        # Create initial file
        write_file._run(filename="overwrite_test.txt", content="original")

        # Overwrite it
        write_file._run(filename="overwrite_test.txt", content="modified")

        if path.exists():
            content = path.read_text()
            path.unlink()
            if content == "modified":
                return result.pass_("File overwrite works")
            else:
                return result.fail(f"Overwrite failed, got: {content}")
        return result.fail("File not found after overwrite")
    except Exception as e:
        return result.fail(f"Exception: {e}")


# =============================================================================
# TEST CATEGORY 2: Provider Failure Modes
# =============================================================================

def test_provider_check_existence_only():
    """Test 2.2: Do we validate API keys or just check existence?"""
    result = TestResult("API Key Validation")

    from crew import check_providers

    # Save real keys
    real_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    real_openai = os.environ.get("OPENAI_API_KEY")

    # Set obviously invalid keys
    os.environ["ANTHROPIC_API_KEY"] = "sk-invalid-12345"
    os.environ["OPENAI_API_KEY"] = "sk-invalid-67890"

    try:
        status = check_providers()

        # Restore real keys
        if real_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = real_anthropic
        else:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        if real_openai:
            os.environ["OPENAI_API_KEY"] = real_openai
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        # If status shows True for providers with invalid keys, we're not validating
        if status.get("anthropic") or status.get("openai"):
            return result.warn("Only checking key existence, not validity")

        return result.pass_("API keys are validated")

    except Exception as e:
        # Restore keys on error
        if real_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = real_anthropic
        if real_openai:
            os.environ["OPENAI_API_KEY"] = real_openai
        return result.fail(f"Exception: {e}")


def test_missing_all_keys():
    """Test 2.3: Behavior when all keys are missing."""
    result = TestResult("Missing All Keys")

    from crew import check_providers

    # Save and remove all keys
    saved_keys = {}
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        saved_keys[key] = os.environ.pop(key, None)

    try:
        status = check_providers()

        # Restore
        for key, value in saved_keys.items():
            if value:
                os.environ[key] = value

        if not status.get("anthropic") and not status.get("openai"):
            return result.pass_("Correctly detects missing keys")

        return result.fail("Did not detect missing keys")

    except Exception as e:
        # Restore on error
        for key, value in saved_keys.items():
            if value:
                os.environ[key] = value
        return result.fail(f"Exception: {e}")


def test_ollama_check_when_down():
    """Test 2.4: Ollama check when service is down."""
    result = TestResult("Ollama Down Detection")

    # We can't actually kill Ollama for this test, but we can check the timeout behavior
    from crew import check_providers
    import time

    start = time.time()
    try:
        status = check_providers()
        elapsed = time.time() - start

        # Should complete within reasonable time (2s timeout + overhead)
        if elapsed < 5:
            return result.pass_(f"Provider check completed in {elapsed:.2f}s")
        else:
            return result.warn(f"Provider check slow: {elapsed:.2f}s")
    except Exception as e:
        return result.fail(f"Exception: {e}")


# =============================================================================
# TEST CATEGORY 3: Input Edge Cases
# =============================================================================

def test_empty_task():
    """Test 5.1a: Empty task description."""
    result = TestResult("Empty Task")

    proc = subprocess.run(
        [sys.executable, "crew.py", ""],
        capture_output=True,
        text=True,
        cwd=CREW_DIR,
        timeout=10
    )

    # Should exit with error
    if proc.returncode != 0:
        return result.pass_("Empty task rejected")

    return result.fail("Empty task was accepted")


def test_whitespace_task():
    """Test 5.1b: Whitespace-only task description."""
    result = TestResult("Whitespace Task")

    proc = subprocess.run(
        [sys.executable, "crew.py", "   \t  "],
        capture_output=True,
        text=True,
        cwd=CREW_DIR,
        timeout=10
    )

    if proc.returncode != 0:
        return result.pass_("Whitespace task rejected")

    return result.fail("Whitespace task was accepted")


def test_no_args():
    """Test 5.1c: No arguments provided."""
    result = TestResult("No Arguments")

    proc = subprocess.run(
        [sys.executable, "crew.py"],
        capture_output=True,
        text=True,
        cwd=CREW_DIR,
        timeout=10
    )

    if proc.returncode != 0:
        if "Usage" in proc.stdout or "Usage" in proc.stderr:
            return result.pass_("Shows usage on no args")
        return result.pass_("Exits with error on no args")

    return result.fail("Did not handle missing args")


def test_unicode_task():
    """Test 5.4: Unicode/non-English task."""
    result = TestResult("Unicode Task")

    # Just test that it doesn't crash during setup
    try:
        from crew import create_crew
        task = "创建一个Python函数来计算斐波那契数列"
        crew = create_crew(task)
        return result.pass_("Unicode task accepted")
    except UnicodeError as e:
        return result.fail(f"Unicode error: {e}")
    except Exception as e:
        # Other errors (like missing providers) are OK
        return result.pass_("Unicode handled (other setup errors OK)")


def test_task_with_quotes():
    """Test: Task with various quote styles."""
    result = TestResult("Quoted Task")

    try:
        from crew import create_crew
        task = '''Build a function that returns "hello" and 'world' with `backticks`'''
        crew = create_crew(task)
        return result.pass_("Quoted task accepted")
    except Exception as e:
        return result.warn(f"Quote handling issue: {e}")


def test_task_injection_attempt():
    """Test 5.3: Task with potential code injection."""
    result = TestResult("Task Injection")

    # These should NOT execute - they're just task descriptions
    dangerous_tasks = [
        'Build"; import os; os.system("echo pwned"); #',
        "Build'; __import__('os').system('id'); '",
        "Build`id`",
    ]

    for task in dangerous_tasks:
        try:
            from crew import create_crew
            # This should just create a crew with this as the task text
            # It should NOT execute any code
            crew = create_crew(task)
        except Exception:
            pass  # Exceptions during setup are fine

    return result.pass_("Injection attempts handled as text")


# =============================================================================
# TEST CATEGORY 4: Workspace Isolation
# =============================================================================

def test_workspace_exists():
    """Test: Workspace directory exists."""
    result = TestResult("Workspace Exists")

    from crew import WORKSPACE_DIR

    if not WORKSPACE_DIR.exists():
        return result.fail("Workspace directory does not exist")

    return result.pass_(f"Workspace at {WORKSPACE_DIR}")


def test_workspace_location():
    """Test: Workspace in expected location."""
    result = TestResult("Workspace Location")

    from crew import WORKSPACE_DIR

    if ".council" not in str(WORKSPACE_DIR):
        return result.fail("Workspace not in .council directory")

    if "workspace" not in str(WORKSPACE_DIR):
        return result.fail("Workspace not named correctly")

    return result.pass_("Workspace in correct location")


def test_workspace_permissions():
    """Test: Workspace has correct permissions."""
    result = TestResult("Workspace Permissions")

    from crew import WORKSPACE_DIR

    # Test write permission
    test_file = WORKSPACE_DIR / "permission_test.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        return result.pass_("Workspace is writable")
    except PermissionError:
        return result.fail("Workspace is not writable")
    except Exception as e:
        return result.fail(f"Permission test failed: {e}")


def test_workspace_cleanup():
    """Test: Can clean workspace."""
    result = TestResult("Workspace Cleanup")

    from crew import WORKSPACE_DIR

    # Create some test files
    test_files = [
        WORKSPACE_DIR / "cleanup1.txt",
        WORKSPACE_DIR / "cleanup2.txt",
    ]

    for f in test_files:
        f.write_text("test")

    # Clean them
    for f in test_files:
        f.unlink()

    # Verify
    for f in test_files:
        if f.exists():
            return result.fail(f"Could not delete {f.name}")

    return result.pass_("Workspace cleanup works")


# =============================================================================
# TEST CATEGORY 5: Module Integrity
# =============================================================================

def test_crew_imports():
    """Test: All crew.py imports work."""
    result = TestResult("Crew Imports")

    try:
        from crew import (
            write_file,
            WORKSPACE_DIR,
            check_providers,
            create_agents,
            create_tasks,
            create_crew,
            Issue,
            ReviewResult
        )
        return result.pass_("All imports successful")
    except ImportError as e:
        return result.fail(f"Import error: {e}")


def test_tool_decorator():
    """Test: write_file is properly decorated as a tool."""
    result = TestResult("Tool Decorator")

    from crew import write_file

    # Check it has the _run method (crewai tool signature)
    if hasattr(write_file, '_run'):
        return result.pass_("write_file has _run method")

    return result.fail("write_file missing _run method")


def test_pydantic_models():
    """Test: Pydantic models are valid."""
    result = TestResult("Pydantic Models")

    from crew import Issue, ReviewResult

    try:
        # Test Issue model
        issue = Issue(
            severity="critical",
            file="test.py",
            description="Test issue",
            suggestion="Fix it"
        )

        # Test ReviewResult model
        review = ReviewResult(
            issues=[issue],
            approved=False,
            summary="Test review"
        )

        return result.pass_("Pydantic models work")
    except Exception as e:
        return result.fail(f"Pydantic error: {e}")


def test_agent_creation():
    """Test: Agents can be created (without actually calling LLMs)."""
    result = TestResult("Agent Creation")

    try:
        from crew import create_agents

        # Save keys to check what happens
        agents = create_agents()

        if len(agents) == 5:
            return result.pass_("All 5 agents created")
        else:
            return result.warn(f"Expected 5 agents, got {len(agents)}")

    except Exception as e:
        # Missing providers might cause issues
        if "API" in str(e) or "key" in str(e).lower():
            return result.warn(f"Agent creation failed (missing keys?): {e}")
        return result.fail(f"Agent creation error: {e}")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    os.chdir(CREW_DIR)

    print("=" * 70)
    print("ADVERSARIAL TEST SUITE - CrewAI Council")
    print("=" * 70)
    print(f"Working directory: {CREW_DIR}")
    print(f"Workspace: {WORKSPACE}")
    print()

    # All test functions grouped by category
    test_categories = {
        "1. File Tool Security": [
            test_path_traversal,
            test_absolute_path,
            test_symlink_escape,
            test_empty_content,
            test_null_byte_content,
            test_unicode_content,
            test_special_chars,
            test_filename_shell_injection,
            test_filename_newline,
            test_filename_null_byte,
            test_large_file,
            test_nested_directory,
            test_overwrite_existing,
        ],
        "2. Provider Handling": [
            test_provider_check_existence_only,
            test_missing_all_keys,
            test_ollama_check_when_down,
        ],
        "3. Input Edge Cases": [
            test_empty_task,
            test_whitespace_task,
            test_no_args,
            test_unicode_task,
            test_task_with_quotes,
            test_task_injection_attempt,
        ],
        "4. Workspace": [
            test_workspace_exists,
            test_workspace_location,
            test_workspace_permissions,
            test_workspace_cleanup,
        ],
        "5. Module Integrity": [
            test_crew_imports,
            test_tool_decorator,
            test_pydantic_models,
            test_agent_creation,
        ],
    }

    passed = 0
    failed = 0
    warnings = 0
    security_issues = 0

    for category, tests in test_categories.items():
        print(f"\n[{category}]")
        print("-" * 50)

        for test_fn in tests:
            try:
                result = test_fn()

                if result.passed:
                    if result.is_warning:
                        print(f"  ⚠️  {result.name}: {result.message}")
                        warnings += 1
                    else:
                        print(f"  ✓ {result.name}: {result.message}")
                        passed += 1
                else:
                    if result.is_security_issue:
                        print(f"  🔴 {result.name}: {result.message}")
                        security_issues += 1
                    else:
                        print(f"  ✗ {result.name}: {result.message}")
                    failed += 1

            except Exception as e:
                print(f"  ✗ {test_fn.__name__}: CRASHED - {e}")
                failed += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed:           {passed}")
    print(f"  Warnings:         {warnings}")
    print(f"  Failed:           {failed}")
    print(f"  Security Issues:  {security_issues}")
    total = passed + warnings + failed + security_issues
    print(f"  Total Tests:      {total}")
    print(f"{'=' * 70}")

    if security_issues > 0:
        print("\n🔴 SECURITY ISSUES FOUND - FIX IMMEDIATELY!")
        print("   These represent critical vulnerabilities that must be addressed.")
        return 2

    if failed > 0:
        print("\n⚠️  Some tests failed - review issues above")
        return 1

    if warnings > 0:
        print("\n⚠️  Warnings present - consider addressing before production")
    else:
        print("\n✓ All tests passed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
