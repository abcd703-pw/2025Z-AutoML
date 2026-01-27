import os
import re
import subprocess
import sys
import ast
import json
from typing import Set, List, Dict, Tuple
from json.decoder import JSONDecodeError

# ==========
# KONFIGURACJA
# ==========
# Pakiety PyPI, których nazwy importowe (moduły) różnią się od nazw pakietów
# Klucz: nazwa modułu do importu, Wartość: nazwa pakietu PyPI
IMPORT_TO_PACKAGE_MAPPING = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
}

# Elementy ignorowane w strukturze katalogów i skanowaniu importów
IGNORE = {'.git', '__pycache__', '.DS_Store', '.gitkeep', '.venv', '.idea'}

# Pakiety, które nie powinny znaleźć się w requirements.txt (narzędzia deweloperskie/podstawowe)
DEV_TOOLS_TO_IGNORE = {
    "pip", "setuptools", "wheel", "pywin32", "colorama", "distribute", "pkg-resources",
}

# ==========
# README UPDATER
# ==========
def tree(dir_path: str, prefix: str = "") -> str:
    """Generuje strukturę katalogów w formacie drzewa."""
    try:
        items = [item for item in os.listdir(dir_path) if item not in IGNORE]
    except FileNotFoundError:
        return ""

    def sort_key(name: str) -> Tuple[int, str]:
        # Zmieniamy item_path na current_item_path, aby uniknąć zacieniania
        current_item_path = os.path.join(dir_path, name)
        if os.path.isdir(current_item_path):
            return 0, name.lower()
        elif name.lower() == "readme.md":
            return 1, name.lower()
        elif name.endswith(".py"):
            return 2, name.lower()
        else:
            return 4, name.lower()

    items = sorted(items, key=sort_key)
    tree_str = ""
    for index, item in enumerate(items):
        # Tutaj nadal używamy item_path (w zakresie funkcji tree)
        item_path = os.path.join(dir_path, item)
        connector = "└─ " if index == len(items) - 1 else "├─ "
        tree_str += f"{prefix}{connector}{item}\n"
        if os.path.isdir(item_path):
            extension = "    " if index == len(items) - 1 else "│   "
            tree_str += tree(item_path, prefix + extension)
    return tree_str


def update_readme() -> None:
    """Aktualizuje sekcję File Structure w pliku README.md."""
    project_dir = "."
    readme_file = "README.md"

    tree_structure = tree(project_dir)
    tree_section = f"## File Structure:\n```\n{tree_structure}```\n"

    try:
        with open(readme_file, "r", encoding="utf-8", errors="ignore") as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = (
            "# Mini_AutoML\n\n"
            "This project was completed as part of the AutoML course at Warsaw University of Technology.\n"
            "The goal of the project is to create a simplified AutoML system that allows automatic execution of a binary classification task on any provided dataset.\n\n"
            "## Authors:\n"
            "- Karol Kacprzak\n"
            "- Ludwik Madej\n"
            "- Mikołaj Bójski\n\n"
        )

    pattern = r"## File Structure:\n```[\s\S]*?```"
    if re.search(pattern, readme_content):
        readme_content = re.sub(pattern, tree_section, readme_content)
    else:
        if not readme_content.endswith("\n"):
            readme_content += "\n"
        readme_content += tree_section

    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("✓ README.md updated with current file structure!")

# ==========
# GIT WRAPPERS
# ==========
def run(cmd: str) -> None:
    """Uruchamia komendę w powłoce i sprawdza błędy."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        print(f"❌ Error running command (Return Code {result.returncode}): {cmd}")
        sys.exit(1)

def sync() -> None:
    """Pobiera zmiany, aktualizuje wymagania i instaluje zależności."""
    print("✓ Pulling latest changes from remote...")
    run("git pull")

    update_requirements()

    print("✓ Installing dependencies...")
    install_cmd = f'"{sys.executable}" -m pip install -r requirements.txt'
    result = subprocess.run(install_cmd, shell=True, check=False)
    if result.returncode != 0:
        print(f"❌ Error installing dependencies (Return Code {result.returncode})!")
        return

    print("✓ Sync completed: repo updated and dependencies installed.")

def git_commit(message: str) -> None:
    """Dodaje wszystkie zmiany i commituje."""
    run("git add .")
    run(f'git commit -m "{message}"')
    print("✓ Changes committed!")

def git_push() -> None:
    """Wysyła commity do zdalnego repozytorium."""
    run("git push")
    print("✓ Changes pushed!")

def git_commit_push(message: str) -> None:
    """Commituje i pushuje."""
    git_commit(message)
    git_push()

# ==========
# REQUIREMENTS UPDATER
# ==========
def _extract_imports_from_ast(node: ast.AST) -> Set[str]:
    """Wspólna funkcja do ekstrakcji importów z węzła AST (unifikuje zduplikowany kod)."""
    imports: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for alias in n.names:
                # Bierzemy tylko główny moduł (np. 'pandas' z 'import pandas.core.frame')
                imports.add(alias.name.split('.')[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module and not n.level: # Ignorujemy względne importy (from . import ...)
                imports.add(n.module.split('.')[0])
    return imports

def get_imports_from_py(filepath: str) -> Set[str]:
    """Zwraca zbiór modułów importowanych w pliku .py"""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            node = ast.parse(f.read(), filename=filepath)
            return _extract_imports_from_ast(node)
    except OSError as e: # Błąd IO/FileNotFound
        print(f"Warning: Could not read {filepath}. Error: {e}")
    except SyntaxError as e: # Niepoprawny kod Pythona
        print(f"Warning: Syntax error in {filepath}. Error: {e}")
    except Exception as e: # Inne błędy parsowania
        print(f"Warning: An unexpected error occurred while parsing {filepath}. Error: {e}")
    return set()

def get_imports_from_ipynb(filepath: str) -> Set[str]:
    """Zwraca zbiór modułów importowanych w notebooku .ipynb"""
    imports: Set[str] = set()
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            notebook = json.load(f)
    except FileNotFoundError as e:
        print(f"Warning: Notebook file not found {filepath}. Error: {e}")
        return imports
    except JSONDecodeError as e: # Precyzyjny wyjątek dla błędów JSON (zamiast ogólnego Exception)
        print(f"Warning: Could not parse JSON in {filepath}. Error: {e}")
        return imports
    except Exception as e:
        print(f"Warning: An unexpected error occurred while loading {filepath}. Error: {e}")
        return imports


    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            try:
                node = ast.parse(source)
                imports.update(_extract_imports_from_ast(node))
            except SyntaxError:
                # Ignorujemy błędy składniowe w pojedynczych komórkach (np. niekompletny kod)
                continue
            except Exception as e:
                print(f"Warning: Error parsing code cell in {filepath}. Error: {e}")

    return imports

def get_project_imports(project_dir: str) -> Set[str]:
    """Skanuje projekt w poszukiwaniu wszystkich importów."""
    imports: Set[str] = set()
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE]

        for f in files:
            filepath = os.path.join(root, f)
            if f in IGNORE:
                continue

            if f.endswith(".py"):
                imports.update(get_imports_from_py(filepath))
            elif f.endswith(".ipynb"):
                imports.update(get_imports_from_ipynb(filepath))
    return imports

def update_requirements(project_dir: str = ".") -> None:
    """Generuje requirements.txt na podstawie importów i zainstalowanych pakietów, filtrując narzędzia deweloperskie."""
    print("✓ Scanning project for imports ...")
    imports = get_project_imports(project_dir)

    # Mapowanie modułów importowanych na nazwy pakietów PyPI
    required_packages: Set[str] = set()
    for imp in imports:
        pkg_name = IMPORT_TO_PACKAGE_MAPPING.get(imp, imp)
        # Usuwamy pakiety deweloperskie/podstawowe
        if pkg_name.lower() not in DEV_TOOLS_TO_IGNORE:
             required_packages.add(pkg_name)

    print(f"Detected core packages: {required_packages}")

    print("✓ Generating requirements.txt ...")
    req_file = os.path.join(project_dir, "requirements.txt")

    # Pobieranie listy zainstalowanych pakietów (z dokładnymi wersjami)
    pip_freeze_cmd = f'"{sys.executable}" -m pip freeze'
    result = subprocess.run(pip_freeze_cmd, shell=True, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"❌ Error running pip freeze: {result.stderr.strip()}")
        return

    installed_packages: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "==" in line:
            name_version = line.strip()
            # Używamy re.split, aby uzyskać samą nazwę pakietu przed separatorem wersji
            match = re.split(r'[=<>!]', name_version, 1)
            if match:
                 name = match[0].strip()
                 installed_packages[name.lower()] = name_version

    package_lines: List[str] = []

    # Dodajemy tylko te pakiety, które są wymagane
    for pkg in sorted(list(required_packages)):
        pkg_match = installed_packages.get(pkg.lower())
        if pkg_match:
            package_lines.append(pkg_match)

    with open(req_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(package_lines) + "\n")

    print(f"✓ requirements.txt updated at {req_file}!")

# ==========
# CLI
# ==========
def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage.py update-readme")
        print("  python manage.py update-requirements")
        print('  python manage.py commit "Message"')
        print("  python manage.py push")
        print('  python manage.py full "Message"')
        print("  python manage.py sync")
        return

    command = sys.argv[1]

    if command == "update-readme":
        update_readme()
    elif command == "update-requirements":
        update_requirements()
    elif command == "commit":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Update"
        git_commit(msg)
    elif command == "push":
        git_push()
    elif command == "full":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Auto update"
        update_readme()
        update_requirements()
        git_commit_push(msg)
    elif command == "sync":
        sync()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()