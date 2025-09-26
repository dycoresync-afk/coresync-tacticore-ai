"""
ID Auditing Module for TactiCore AI
- Monitors ID records (e.g., in CSV/JSON) for tampering by hash comparison.
- Alerts on discrepancies (console output; extendable to email/SMS).
- Run-able standalone: python id_auditor.py --file ids.json --stored_hash [HASH]
- Requirements: pip install watchdog
"""

import os
import time
import hashlib
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime

class IDAuditor:
    def __init__(self, file_path: str, stored_hash: str):
        self.file_path = file_path
        self.stored_hash = bytes.fromhex(stored_hash)
        self.current_hash = None
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        print(f"Initial stored hash: {self.stored_hash.hex()}")

    def compute_current_hash(self) -> bytes:
        """Compute SHA-256 hash of current ID records file."""
        with open(self.file_path, 'rb') as f:
            content = f.read()
        hash_object = hashlib.sha256(content)
        self.current_hash = hash_object.digest()
        return self.current_hash

    def audit_ids(self) -> bool:
        """Audit ID records for integrity."""
        self.compute_current_hash()
        is_intact = self.current_hash == self.stored_hash
        if not is_intact:
            self._alert_tamper()
        return is_intact

    def _alert_tamper(self):
        """Alert on tamper detection (console; extendable to email/SMS)."""
        message = f"ALERT: Tamper attempt detected on {self.file_path}!\nStored hash: {self.stored_hash.hex()}\nCurrent hash: {self.current_hash.hex()}\nTimestamp: {datetime.now().isoformat()}"
        print(message)
        # Extend: Send email/SMS (e.g., via smtplib or Twilio)
        # import smtplib
        # ... (add email logic here)

class IDEventHandler(FileSystemEventHandler):
    def __init__(self, auditor: IDAuditor):
        self.auditor = auditor

    def on_modified(self, event):
        if event.src_path == self.auditor.file_path:
            print(f"ID record file modified: {event.src_path}")
            if not self.auditor.audit_ids():
                print("Tamper detected - alert triggered!")

# Example ID records file (ids.json) format:
# [
#     {"id": "USER001", "name": "John Doe", "role": "Admin", "timestamp": "2025-09-26T09:00:00Z"},
#     {"id": "DOC001", "title": "Mission Plan", "version": "1.0", "timestamp": "2025-09-26T09:30:00Z"}
# ]

# Standalone Run-able Example
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="ID Auditor")
    parser.add_argument('--file', required=True, help="Path to ID records file (JSON/CSV)")
    parser.add_argument('--stored_hash', required=True, help="Stored hash of original ID records")
    args = parser.parse_args()
    
    auditor = IDAuditor(args.file, args.stored_hash)
    
    # Initial audit
    if auditor.audit_ids():
        print("Initial ID audit passed.")
    else:
        print("Initial ID audit failed - records tampered!")

    # Start monitoring
    observer = Observer()
    observer.schedule(IDEventHandler(auditor), path=os.path.dirname(args.file), recursive=False)
    observer.start()
    
    print(f"Monitoring started at 09:58 PM KST, September 26, 2025. Modify the file to test ID tampering.")
    try:
        while True:
            time.sleep(1)
            auditor.audit_ids()  # Periodical check (e.g., every second)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
