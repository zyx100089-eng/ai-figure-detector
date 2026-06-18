"""
Safely execute LLM-generated matplotlib code and capture the output figure.
Runs each script in a subprocess with a timeout to prevent hangs.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path


def execute_code(code: str, output_path: str, timeout: int = 30) -> bool:
    """
    Execute matplotlib code in a subprocess and save the output figure.
    Returns True if a figure was successfully generated.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "generate.py")
        tmp_output = os.path.join(tmpdir, "output.png")

        # Prepend matplotlib backend setting to avoid display issues
        full_code = "import matplotlib\nmatplotlib.use('Agg')\n" + code

        with open(script_path, "w") as f:
            f.write(full_code)

        try:
            result = subprocess.run(
                ["python", script_path],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                print(f"  Code execution failed: {result.stderr[:200]}")
                return False

            if os.path.exists(tmp_output):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(tmp_output, output_path)
                return True

            # Check if code saved to a different filename
            png_files = list(Path(tmpdir).glob("*.png"))
            if png_files:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(str(png_files[0]), output_path)
                return True

            print("  No output image generated")
            return False

        except subprocess.TimeoutExpired:
            print("  Code execution timed out")
            return False
        except Exception as e:
            print(f"  Execution error: {e}")
            return False


def extract_code_from_response(response: str) -> str | None:
    """Extract Python code from an LLM response (handles markdown code blocks)."""
    # Try to find ```python ... ``` blocks
    if "```python" in response:
        parts = response.split("```python")
        if len(parts) > 1:
            code = parts[1].split("```")[0]
            return code.strip()

    # Try to find ``` ... ``` blocks
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("\n"):
                code = code[1:]
            return code.strip()

    # If no code blocks, check if the whole response looks like code
    if "import " in response and ("plt." in response or "matplotlib" in response):
        return response.strip()

    return None
