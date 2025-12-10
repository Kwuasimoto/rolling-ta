import sys
import re
import os

def generate_slug(title):
    # Basic GitHub slug generation
    slug = title.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    return slug

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if TOC already exists
    if "## Table of Contents" in content:
        print(f"Skipping {filepath}: TOC already exists")
        return

    lines = content.split('\n')
    
    # Simple state machine to ignore code blocks
    in_code_block = False
    
    headers = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            continue
            
        # Match headers H2 to H6. H1 is usually title, so exclude it from TOC unless desired.
        # User request: "assists in the LLM's indexing". Usually H2+ is best.
        match = re.match(r'^(#{2,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append((level, title))

    if not headers:
        print(f"No headers (H2-H6) found in {filepath}")
        return

    toc_lines = []
    toc_lines.append("## Table of Contents")
    toc_lines.append("")
    
    # Normalize indentation
    if headers:
        min_level = min(h[0] for h in headers)
        for level, title in headers:
            indent = "  " * (level - min_level)
            slug = generate_slug(title)
            toc_lines.append(f"{indent}- [{title}](#{slug})")
    
    toc_lines.append("")
    toc_content = "\n".join(toc_lines)
    
    # Insertion logic
    # Check for frontmatter
    insert_idx = 0
    if lines and lines[0].strip() == '---':
        # Find end of frontmatter
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                insert_idx = i + 1
                break
    
    # Insert TOC
    # Ensure some spacing
    if insert_idx > 0 and lines[insert_idx].strip() == "":
        pass # Already has a newline
    else:
        toc_content += "\n"

    new_content_lines = lines[:insert_idx] + [toc_content] + lines[insert_idx:]
    new_content = "\n".join(new_content_lines)

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Has added TOC to {filepath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 add_toc.py <file>")
        sys.exit(1)
    
    process_file(sys.argv[1])
