#!/usr/bin/env python3
import sys
import os
import requests
import re
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# We can't easily import from scripts/steps/step_2_1... because it's not a module. 
# I'll replicate the get_auth and list_remote_files logic here for testing.

def get_auth():
    import netrc
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return (auth[0], auth[2])
    except Exception:
        pass
    user = os.getenv("CDDIS_USER")
    passwd = os.getenv("CDDIS_PASS")
    if user and passwd:
        return (user, passwd)
    return None

def list_remote_files(session, url):
    print(f"Listing {url} ...")
    try:
        resp = session.get(url, timeout=30)
        print(f"Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"Error: {resp.text[:200]}")
            return []
        
        # Check for HTML
        if "<html" in resp.text.lower() or "<!doctype" in resp.text.lower():
            print("Detected HTML response (likely directory index or login page)")
            # Try to parse links from HTML if it's an index
            # This handles standard Apache/nginx indexes
            lines = resp.text.splitlines()
            files = []
            for line in lines:
                # Naive regex for href
                m = re.search(r'href="([^"]+)"', line)
                if m:
                    name = m.group(1)
                    # skip parent directory, query params, etc
                    if name in ['../', './'] or name.startswith('?') or name.startswith('/'):
                        continue
                    files.append(name.rstrip('/'))
            print(f"Parsed {len(files)} files/dirs from HTML")
            return files
            
        # Text listing (if ?list works)
        return resp.text.splitlines()
    except Exception as e:
        print(f"Exception: {e}")
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage: probe_cddis.py <url>")
        sys.exit(1)
        
    url = sys.argv[1]
    
    auth = get_auth()
    if not auth:
        print("No auth found")
        sys.exit(1)
        
    s = requests.Session()
    s.auth = auth
    
    # Try ?list first
    files = list_remote_files(s, url + "?list")
    if not files:
        print("Trying without ?list (HTML scrape)...")
        files = list_remote_files(s, url)
        
    print("Files found:")
    for f in files[:20]:
        print(f" - {f}")
        
if __name__ == "__main__":
    main()
