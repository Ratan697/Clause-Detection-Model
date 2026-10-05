import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Removes common document noise and normalizes whitespace."""
    # Remove standalone page numbers (e.g., "Page 12 of 45" or "- 3 -")
    text = re.sub(r'(?i)\bpage\s+\d+(\s+of\s+\d+)?\b', '', text)
    text = re.sub(r'^\s*[-—–]\s*\d+\s*[-—–]\s*$', '', text, flags=re.MULTILINE)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def segment_contract_into_clauses(full_text: str) -> List[Dict[str, str]]:
    """
    Intelligently splits a full contract into individual provisions/clauses.
    Detects:
    - Section headers (e.g. "Section 1.", "1.1", "Article IV", "10.")
    - Standard paragraph breaks
    - Title / clause names
    """
    cleaned = clean_text(full_text)
    if not cleaned:
        return []

    # Regex patterns for section boundaries
    section_split_pattern = r'(?m)(?=^(?:SECTION|ARTICLE|CLAUSE|\d+\.|\([a-z0-9]+\))\s+[A-Z0-9])'
    
    raw_chunks = re.split(section_split_pattern, cleaned)
    
    # If the document wasn't cleanly divided by formal Section tags, fall back to double-newline paragraphs
    if len(raw_chunks) <= 2:
        raw_chunks = cleaned.split('\n\n')
        
    clauses = []
    clause_counter = 1
    
    for chunk in raw_chunks:
        chunk_clean = chunk.strip()
        words = chunk_clean.split()
        
        # Filter out trivial fragments (less than 6 words, unless it's a section title)
        if len(words) < 6:
            continue
            
        # Extract probable clause heading / title if present
        first_line = chunk_clean.split('\n')[0].strip()
        heading_match = re.match(r'^(?:(?:SECTION|ARTICLE|CLAUSE|\d+\.?|\([a-z0-9]+\))\s*[:\-–\.]?\s*)([A-Za-z\s,/&\-]+)', first_line)
        
        if heading_match and len(heading_match.group(1).split()) <= 6:
            clause_heading = heading_match.group(0).strip(" :.-")
        else:
            # Fallback to first 5 words as summary title
            clause_heading = f"Clause #{clause_counter}"
            
        # Clean multi-line breaks inside the clause text
        single_line_text = re.sub(r'\s+', ' ', chunk_clean).strip()
        
        clauses.append({
            "id": clause_counter,
            "heading": clause_heading,
            "text": single_line_text,
            "word_count": len(words)
        })
        clause_counter += 1
        
    # If all segmentation attempts resulted in 0 clauses, wrap the whole text
    if not clauses and len(cleaned.split()) >= 5:
        clauses.append({
            "id": 1,
            "heading": "Full Provision",
            "text": re.sub(r'\s+', ' ', cleaned).strip(),
            "word_count": len(cleaned.split())
        })
        
    return clauses

