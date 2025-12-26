import os

# Define the file structure
files = [
    ".github/workflows/ci.yml",
    "configs/coco80_labels.txt",
    "configs/defaults.yaml",
    "src/promptgroundboxbench/__init__.py",
    "src/promptgroundboxbench/benchmarks/__init__.py",
    "src/promptgroundboxbench/benchmarks/speed.py",
    "src/promptgroundboxbench/cli.py",
    "src/promptgroundboxbench/config.py",
    "src/promptgroundboxbench/demo/__init__.py",
    "src/promptgroundboxbench/demo/gradio_app.py",
    "src/promptgroundboxbench/engines/__init__.py",
    "src/promptgroundboxbench/engines/base.py",
    "src/promptgroundboxbench/engines/grounding_dino.py",
    "src/promptgroundboxbench/engines/yolo.py",
    "src/promptgroundboxbench/eval/__init__.py",
    "src/promptgroundboxbench/eval/coco.py",
    "src/promptgroundboxbench/types.py",
    "src/promptgroundboxbench/utils/__init__.py",
    "src/promptgroundboxbench/utils/draw.py",
    "src/promptgroundboxbench/utils/image.py",
    "src/promptgroundboxbench/utils/io.py",
    "src/promptgroundboxbench/utils/prompt.py",
    "src/promptgroundboxbench/utils/timing.py",
    "tests/test_prompt.py",
    "tests/test_types.py",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "environment.yml",
    "pyproject.toml",
]

# Define .gitignore content
gitignore_content = """# Python
__pycache__/
*.py[cod]
*.pyd
*.so
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Envs
.venv/
venv/
.env
.env.*

# Builds
build/
dist/
*.egg-info/

# OS / IDE
.DS_Store
Thumbs.db
.idea/
.vscode/

# Outputs
runs/
reports/
"""

def create_structure():
    # Create .gitignore
    print("Creating .gitignore...")
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)

    # Create other files
    for file_path in files:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        if not os.path.exists(file_path):
            print(f"Creating {file_path}...")
            with open(file_path, "w") as f:
                pass  # Create empty file
        else:
            print(f"Skipping {file_path} (already exists)")

if __name__ == "__main__":
    create_structure()
    print("\nProject structure generated successfully.")
