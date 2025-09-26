"""
Real-Time Tamper Detection Monitor for TactiCore AI
- Monitors a file or directory for changes using watchdog.
- Computes SHA-256 hash and compares with stored hash to detect tampering.
- Logs events and alerts on detection (console; extendable to email/SMS).
- Run-able standalone: python tamper_monitor.py --path /path/to/file --stored_hash [HASH]
- Requirements: pip install watchdog
"""

import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import argparse

class TamperMonitor:
    def __init__(self, path: str, stored_hash: str):
        self.path = os.path.abspath(path)
        self.stored_hash = bytes.fromhex(stored_hash)
        self.last_hash = None
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path {self.path} not found.")
        print(f"Monitoring started at {datetime.now().strftime('%I:%M %p KST, %B %d, %Y')} KST")
        print(f"Initial stored hash: {self.stored_hash.hex()}")

    def compute_hash(self) -> bytes:
        """Compute SHA-256 hash of the file content."""
        with open(self.path, 'rb') as f:
            content = f.read()
        hash_object = hashlib.sha256(content)
        return hash_object.digest()

    def check_integrity(self) -> bool:
        """Check file integrity by comparing current hash with stored hash."""
        current_hash = self.compute_hash()
        self.last_hash = current_hash
        is_intact = current_hash == self.stored_hash
        if not is_intact:
            self._alert_tamper(current_hash.hex())
        return is_intact

    def _alert_tamper(self, current_hash: str):
        """Alert on tamper detection (console; extendable to email/SMS)."""
        timestamp = datetime.now().strftime('%I:%M %p KST, %B %d, %Y')
        message = f"ALERT: Tamper detected in {os.path.basename(self.path)} at {timestamp}!\nStored hash: {self.stored_hash.hex()}\nCurrent hash: {current_hash}"
        print(message)
        # Extend: Send email/SMS (e.g., via smtplib or Twilio)
        # import smtplib
        # ... (add email logic here)

class TamperEventHandler(FileSystemEventHandler):
    def __init__(self, monitor: TamperMonitor):
        self.monitor = monitor

    def on_modified(self, event):
        """Handle file modification events."""
        if event.src_path == self.monitor.path:
            print(f"File modified: {os.path.basename(event.src_path)} at {datetime.now().strftime('%I:%M %p KST')}")
            if not self.monitor.check_integrity():
                print("Tamper alert triggered!")

    def on_created(self, event):
        """Handle file creation events (optional monitoring extension)."""
        if event.src_path == self.monitor.path:
            print(f"New file detected: {os.path.basename(event.src_path)} at {datetime.now().strftime('%I:%M %p KST')}")
            self.monitor.check_integrity()

    def on_deleted(self, event):
        """Handle file deletion events (optional monitoring extension)."""
        if event.src_path == self.monitor.path:
            print(f"File deleted: {os.path.basename(event.src_path)} at {datetime.now().strftime('%I:%M %p KST')}")
            # Note: Cannot check hash on deletion; log only.

# Standalone Run-able Example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Tamper Detection Monitor")
    parser.add_argument('--path', required=True, help="Path to file or directory to monitor")
    parser.add_argument('--stored_hash', required=True, help="Stored hash of original file (e.g., SHA-256)")
    args = parser.parse_args()

    monitor = TamperMonitor(args.path, args.stored_hash)
    
    # Initial integrity check
    if monitor.check_integrity():
        print("Initial integrity check passed.")
    else:
        print("Initial integrity check failed - file tampered!")

    # Start real-time monitoring
    event_handler = TamperEventHandler(monitor)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(args.path) or '.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive for observer
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
