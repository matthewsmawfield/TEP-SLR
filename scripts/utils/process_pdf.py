#!/usr/bin/env python3
"""
Unified PDF Processing Script for TEP-SLR
Compresses PDF and embeds comprehensive metadata in one operation.
Customized for: Paper 5 - Satellite Laser Ranging (Mombasa v0.1)

Usage:
    python scripts/utils/process_pdf.py <input_pdf> [--quality ebook|printer|prepress|default]
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import tempfile

def compress_pdf(input_path, output_path, quality='ebook'):
    """Compress PDF using Ghostscript."""
    quality_settings = {
        'screen': '/screen',      # 72 dpi
        'ebook': '/ebook',        # 150 dpi
        'printer': '/printer',    # 300 dpi
        'prepress': '/prepress',  # 300 dpi, color preserving
        'default': '/default'
    }
    
    if quality not in quality_settings:
        raise ValueError(f"Quality must be one of: {', '.join(quality_settings.keys())}")
    
    gs_quality = quality_settings[quality]
    
    # Get original size
    original_size = os.path.getsize(input_path)
    
    # Compress using Ghostscript
    cmd = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS={gs_quality}',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        input_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        compressed_size = os.path.getsize(output_path)
        reduction = ((original_size - compressed_size) / original_size) * 100
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'compressed_mb': compressed_size / (1024 * 1024),
            'reduction_pct': reduction
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ghostscript compression failed: {e.stderr.decode()}")

def embed_metadata(pdf_path, metadata):
    """Embed metadata into PDF using exiftool."""
    cmd = ['exiftool']
    
    # Add all metadata fields
    for key, value in metadata.items():
        cmd.extend([f'-{key}={value}'])
    
    # Overwrite original
    cmd.extend(['-overwrite_original', pdf_path])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to Ghostscript pdfmark if exiftool is missing
        print("  ⚠ exiftool failed or not found, falling back to Ghostscript for metadata...")
        return embed_metadata_gs(pdf_path, metadata)

def embed_metadata_gs(pdf_path, metadata):
    """Fallback metadata embedding using Ghostscript pdfmark."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ps', delete=False) as f:
        meta_str = ""
        for key, value in metadata.items():
            meta_str += f"/{key} ({value}) "
        f.write(f"[ {meta_str} /DOCINFO pdfmark")
        pdfmark_path = f.name

    output_path = f"{pdf_path}.tmp"
    cmd = [
        'gs', '-sDEVICE=pdfwrite', '-dNOPAUSE', '-dBATCH', '-dQUIET',
        f'-sOutputFile={output_path}', pdf_path, pdfmark_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(output_path, pdf_path)
        return True
    finally:
        if os.path.exists(pdfmark_path):
            os.unlink(pdfmark_path)

def verify_metadata(pdf_path, expected_fields):
    """Verify metadata was embedded correctly."""
    cmd = ['exiftool'] + [f'-{field}' for field in expected_fields] + [pdf_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Compress TEP-SLR PDF and embed metadata in one operation'
    )
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument(
        '--quality',
        choices=['screen', 'ebook', 'printer', 'prepress', 'default'],
        default='ebook',
        help='Compression quality (default: ebook)'
    )
    parser.add_argument(
        '--doi',
        default='10.5281/zenodo.18064582',
        help='DOI to embed in metadata'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_pdf).resolve()
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing TEP-SLR: {input_path}")
    print(f"Quality: {args.quality}")
    print()
    
    # Step 1: Compress PDF
    print("Step 1: Compressing PDF...")
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        stats = compress_pdf(str(input_path), tmp_path, args.quality)
        os.replace(tmp_path, str(input_path))
        
        print(f"  Original:    {stats['original_mb']:.2f} MB")
        print(f"  Compressed:  {stats['compressed_mb']:.2f} MB")
        print(f"  Reduction:   {stats['reduction_pct']:.1f}%")
        print()
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"Error during compression: {e}")
        sys.exit(1)
    
    # Step 2: Embed metadata
    print("Step 2: Embedding TEP-SLR metadata...")
    
    metadata = {
        'Title': 'Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging',
        'Author': 'Matthew Lukin Smawfield',
        'Subject': f'This study presents an independent optical-domain validation of the Temporal Equivalence Principle (TEP) using 11 years (2015–2025) of high-precision Satellite Laser Ranging (SLR) data from the International Laser Ranging Service (ILRS). Analysis of 192,561 residuals from LAGEOS-1/2 and Etalon-1/2 reveals three signatures consistent with TEP\'s conformal sector: path-length dependent coherence decay, spectral power concentration in the predicted TEP band (10–500 μHz), and frequency independence between optical and microwave domains. The results demonstrate that the distance-structured correlations previously identified in GNSS atomic clock networks (Paper 1-3) are not artifacts of microwave processing or clock systematics, but reflect a fundamental property of spacetime propagation. DOI: {args.doi}',
        'Keywords': 'Temporal Equivalence Principle; TEP-SLR; satellite laser ranging; SLR; LAGEOS; optical validation; conformal sector; geodesy; relativity; Proper Time; Global Time Echoes; Mombasa v0.1',
        'Creator': 'Matthew Lukin Smawfield',
        'Producer': 'TEP-SLR Research Project',
        'Copyright': 'Creative Commons Attribution 4.0 International License (CC BY 4.0)',
        'CreationDate': '2025:12:30 00:00:00',
        'ModifyDate': '2025:12:30 00:00:00'
    }
    
    try:
        embed_metadata(str(input_path), metadata)
        print("  Metadata embedded successfully")
        print()
        
    except Exception as e:
        print(f"Error during metadata embedding: {e}")
        sys.exit(1)
    
    # Step 3: Verify
    print("Step 3: Verifying metadata...")
    verification = verify_metadata(
        str(input_path),
        ['Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Copyright']
    )
    
    if verification:
        print("  ✓ Metadata verified")
        print()
        print("Verification output:")
        print(verification)
    else:
        print("  ⚠ Could not verify metadata via exiftool (may still be embedded via GS)")
    
    print()
    print(f"✓ Processing complete: {input_path}")
    print(f"  Final size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()
