#!/usr/bin/env python3
"""
Remove all non-ASCII characters and emojis from project files.
Preserves LaTeX math notation which is ASCII-based.
"""

import json
import re
from pathlib import Path

def remove_non_ascii_from_text(text):
    """Remove non-ASCII characters while preserving LaTeX."""
    # Replace common emojis and non-ASCII with ASCII equivalents
    result = ""
    for char in text:
        if ord(char) < 128:
            # ASCII character - keep it
            result += char
        else:
            # Non-ASCII - check for common replacements
            # Most emojis will be skipped, some have ASCII equivalents
            pass  # Skip non-ASCII
    return result

def clean_notebook(path):
    """Remove non-ASCII from Jupyter notebook."""
    print(f"Cleaning notebook: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if 'source' in cell:
            if isinstance(cell['source'], list):
                cell['source'] = [remove_non_ascii_from_text(line) for line in cell['source']]
            else:
                cell['source'] = remove_non_ascii_from_text(cell['source'])
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    
    print(f"  ✓ Notebook cleaned")

def clean_markdown(path):
    """Remove non-ASCII from Markdown file."""
    print(f"Cleaning markdown: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = remove_non_ascii_from_text(content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"  ✓ Markdown cleaned")

def clean_text_file(path):
    """Remove non-ASCII from plain text file."""
    print(f"Cleaning text file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = remove_non_ascii_from_text(content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"  ✓ Text file cleaned")

def main():
    repo_root = Path.cwd()
    
    print("=" * 60)
    print("Removing non-ASCII characters and emojis from all files")
    print("=" * 60)
    print()
    
    # Process Jupyter notebooks
    for nb_file in repo_root.rglob("*.ipynb"):
        try:
            clean_notebook(nb_file)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Process Markdown files
    for md_file in repo_root.rglob("*.md"):
        try:
            clean_markdown(md_file)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Process text files
    for txt_file in repo_root.rglob("*.txt"):
        try:
            clean_text_file(txt_file)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print()
    print("=" * 60)
    print("Cleaning complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
