"""
Module: data_fetch.py
Objective: Acquire genuine positive ARG sequences from the official CARD database
and candidate negative non-ARG bacterial sequences from UniProtKB/Swiss-Prot.
Maintains complete provenance documentation (source URL, retrieval timestamp,
database version, query parameters, checksums).
"""

import os
import sys
import json
import tarfile
import hashlib
import datetime
from pathlib import Path
import requests

CARD_LATEST_DATA_URL = "https://card.mcmaster.ca/latest/data"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

DATA_RAW_DIR = Path("data/raw")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_card_data(output_fasta_path: Path = DATA_RAW_DIR / "card_homolog_raw.fasta") -> dict:
    """
    Download the latest CARD database archive and extract the Protein Homolog Model FASTA.
    
    The Protein Homolog Model represents canonical ARGs (functional enzymes/pumps)
    and excludes target mutation variant models (which differ by only a single point mutation).
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_RAW_DIR / "card-data.tar.bz2"
    
    print(f"[CARD] Connecting to {CARD_LATEST_DATA_URL}...")
    headers = {"User-Agent": "Bioinformatics-ARG-Classifier/1.0 (Academic Research; Student Project)"}
    
    response = requests.get(CARD_LATEST_DATA_URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    
    total_bytes = 0
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
                
    print(f"[CARD] Download complete: {total_bytes / (1024*1024):.2f} MB saved to {archive_path}")
    archive_hash = compute_sha256(archive_path)
    
    # Extract protein_fasta_protein_homolog_model.fasta and card.json
    extracted_card_version = "Unknown"
    with tarfile.open(archive_path, "r:bz2") as tar:
        members = tar.getnames()
        # Look for the homolog model fasta
        target_name = None
        for name in members:
            if "protein_fasta_protein_homolog_model.fasta" in name:
                target_name = name
                break
        if not target_name:
            raise RuntimeError(f"Could not find protein_fasta_protein_homolog_model.fasta in archive. Available: {members[:10]}")
            
        print(f"[CARD] Extracting {target_name} to {output_fasta_path}...")
        extracted_file = tar.extractfile(target_name)
        with open(output_fasta_path, "wb") as out_f:
            out_f.write(extracted_file.read())
            
        # Try to read card.json or version info
        for name in members:
            if name.endswith("card.json"):
                try:
                    card_json_data = json.loads(tar.extractfile(name).read().decode("utf-8", errors="ignore"))
                    extracted_card_version = card_json_data.get("_version", "Unknown")
                except Exception:
                    pass
                break

    fasta_hash = compute_sha256(output_fasta_path)
    # Count sequences
    seq_count = 0
    with open(output_fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                seq_count += 1
                
    print(f"[CARD] Successfully extracted {seq_count} positive ARG sequences (Version: {extracted_card_version}).")
    
    return {
        "source": "Comprehensive Antibiotic Resistance Database (CARD)",
        "source_url": CARD_LATEST_DATA_URL,
        "retrieval_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database_version": extracted_card_version,
        "model_type": "Protein Homolog Model (canonical resistance genes; excludes single point mutation models)",
        "archive_file": str(archive_path),
        "archive_sha256": archive_hash,
        "extracted_file": str(output_fasta_path),
        "extracted_sha256": fasta_hash,
        "raw_sequence_count": seq_count
    }


def fetch_uniprot_negative_data(
    target_count: int = 5000,
    output_fasta_path: Path = DATA_RAW_DIR / "uniprot_negative_raw.fasta"
) -> dict:
    """
    Fetch candidate non-ARG bacterial protein sequences from UniProtKB/Swiss-Prot (reviewed).
    
    Selection and Exclusion Criteria:
    - Reviewed: true (Swiss-Prot, high manual curation and annotation confidence)
    - Taxonomy: Bacteria (taxonomy_id:2)
    - Strict Exclusion of resistance keywords:
      * KW-0046 (Antibiotic resistance)
      * KW-0089 (Beta-lactamase)
    - Strict Exclusion of resistance Gene Ontology (GO) terms:
      * GO:0046677 (response to antibiotic)
      * GO:0016999 (antibiotic metabolic process)
      * GO:0042493 (response to drug)
      * GO:0009249 (protein repair / multidrug efflux)
    - Additional textual negative filters:
      Excludes proteins matching *lactamase*, *efflux*, *aminoglycoside*, *tetracycline*, *macrolide*.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    query = (
        'reviewed:true AND taxonomy_id:2 '
        'NOT keyword:KW-0046 '
        'NOT keyword:KW-0089 '
        'NOT go:0046677 '
        'NOT go:0016999 '
        'NOT go:0042493 '
        'NOT go:0009249'
    )
    
    print(f"[UniProt] Querying Swiss-Prot for reviewed non-ARG bacterial sequences...")
    print(f"[UniProt] Query: {query}")
    
    page_size = 500
    headers = {"User-Agent": "Bioinformatics-ARG-Classifier/1.0 (Academic Research; Student Project)"}
    
    downloaded_records = []
    current_url = UNIPROT_SEARCH_URL
    params = {
        "query": query,
        "format": "fasta",
        "size": page_size
    }
    
    session = requests.Session()
    total_fetched = 0
    
    with open(output_fasta_path, "w", encoding="utf-8") as out_f:
        while current_url and total_fetched < target_count:
            req_params = params if current_url == UNIPROT_SEARCH_URL else None
            response = session.get(current_url, headers=headers, params=req_params, timeout=60)
            response.raise_for_status()
            
            fasta_chunk = response.text
            if not fasta_chunk.strip():
                break
                
            # Count headers in chunk
            chunk_headers = [line for line in fasta_chunk.splitlines() if line.startswith(">")]
            num_in_chunk = len(chunk_headers)
            
            # Additional safety regex filter for header descriptions
            # Exclude any entry that mentions resistance-related terms in description
            lines_to_write = []
            skip_current = False
            for line in fasta_chunk.splitlines():
                if line.startswith(">"):
                    lower_line = line.lower()
                    if any(term in lower_line for term in ["lactamase", "efflux", "aminoglycoside", "tetracycline resistance", "macrolide resistance"]):
                        skip_current = True
                    else:
                        skip_current = False
                        lines_to_write.append(line)
                        total_fetched += 1
                else:
                    if not skip_current:
                        lines_to_write.append(line)
                        
                if total_fetched >= target_count:
                    break
                    
            out_f.write("\n".join(lines_to_write) + "\n")
            print(f"[UniProt] Fetched {total_fetched}/{target_count} sequences...")
            
            # Check pagination via Link header
            link_header = response.headers.get("Link")
            if link_header and 'rel="next"' in link_header:
                # Extract next url: <https://...>; rel="next"
                parts = link_header.split(";")
                next_url = parts[0].strip("<> ")
                current_url = next_url
            else:
                current_url = None
                
    fasta_hash = compute_sha256(output_fasta_path)
    print(f"[UniProt] Completed fetching {total_fetched} non-ARG sequences saved to {output_fasta_path}")
    
    return {
        "source": "UniProtKB / Swiss-Prot (manually reviewed)",
        "source_url": UNIPROT_SEARCH_URL,
        "retrieval_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database_release": "UniProtKB current release",
        "selection_criteria": (
            "Reviewed bacterial proteins (Swiss-Prot, taxonomy_id:2). "
            "Explicit exclusion of KW-0046 (Antibiotic resistance), KW-0089 (Beta-lactamase), "
            "GO:0046677, GO:0016999, GO:0042493, GO:0009249, and resistance terms in protein descriptions."
        ),
        "query_string": query,
        "downloaded_file": str(output_fasta_path),
        "sha256": fasta_hash,
        "raw_sequence_count": total_fetched
    }


def run_data_acquisition(target_negative_count: int = 5000):
    """Orchestrate data acquisition for both positive (CARD) and negative (UniProt) sets."""
    print("=" * 70)
    print("PHASE 2: REAL BIOLOGICAL DATA ACQUISITION")
    print("=" * 70)
    
    card_meta = fetch_card_data()
    uniprot_meta = fetch_uniprot_negative_data(target_count=target_negative_count)
    
    provenance = {
        "project": "Machine Learning-Based Binary Classification of ARGs vs Non-ARGs",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "positive_dataset": card_meta,
        "negative_dataset": uniprot_meta
    }
    
    provenance_path = DATA_RAW_DIR / "data_provenance.json"
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
        
    print(f"[Provenance] Metadata successfully recorded at {provenance_path}")
    print("=" * 70)
    return provenance


if __name__ == "__main__":
    count = 5000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    run_data_acquisition(target_negative_count=count)
