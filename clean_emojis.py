#!/usr/bin/env python3
"""
Remove emojis and common non-ASCII characters from project files.
Preserves mathematical notation, em-dashes, and other important Unicode.
"""

import json
import re
import unicodedata
from pathlib import Path

def is_emoji(char):
    """Check if character is an emoji."""
    # Emoji Unicode ranges
    emoji_ranges = [
        (0x1F300, 0x1F9FF),  # Miscellaneous Symbols and Pictographs, Emoticons, etc.
        (0x2600, 0x27BF),    # Miscellaneous Symbols
        (0x2702, 0x27B0),    # Dingbats
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x2300, 0x23FF),    # Miscellaneous Technical
        (0x2B50, 0x2B55),    # Star symbols
    ]
    
    code = ord(char)
    for start, end in emoji_ranges:
        if start <= code <= end:
            return True
    
    return False

def remove_emojis_only(text):
    """Remove only emojis, preserve mathematical symbols and dashes."""
    result = ""
    for char in text:
        if is_emoji(char):
            # Skip emoji - replace with space if between words
            if result and result[-1] not in (' ', '\n', '-'):
                result += ' '
        else:
            result += char
    
    # Clean up multiple spaces
    result = re.sub(r' +', ' ', result)
    # Remove space before punctuation
    result = re.sub(r' ([.,!?;:\)])', r'\1', result)
    
    return result

def clean_notebook(path):
    """Remove emojis from Jupyter notebook."""
    print(f"Cleaning notebook: {path.name}")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if 'source' in cell:
            if isinstance(cell['source'], list):
                cell['source'] = [remove_emojis_only(line) for line in cell['source']]
            else:
                cell['source'] = remove_emojis_only(cell['source'])
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"  OK")

def clean_markdown(path):
    """Remove emojis from Markdown file."""
    print(f"Cleaning markdown: {path.name}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = remove_emojis_only(content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"  OK")

def clean_text_file(path):
    """Remove emojis from plain text file."""
    print(f"Cleaning text file: {path.name}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = remove_emojis_only(content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"  OK")

def main():
    repo_root = Path.cwd()
    
    print("=" * 60)
    print("Removing emojis from all files (preserving Unicode math)")
    print("=" * 60)
    print()
    
    # Process Jupyter notebooks
    for nb_file in sorted(repo_root.rglob("*.ipynb")):
        try:
            clean_notebook(nb_file)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Process Markdown files
    for md_file in sorted(repo_root.rglob("*.md")):
        try:
            clean_markdown(md_file)
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Process text files
    for txt_file in sorted(repo_root.rglob("*.txt")):
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
