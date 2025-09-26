"""
Real-Time Tamper Detection Monitor with Multi-Channel Alerts for TactiCore AI
- Monitors a file for changes using watchdog.
- Computes SHA-256 hash and compares with stored hash to detect tampering.
- Sends alerts via email, SMS, and mobile push on tamper detection.
- Run-able standalone: python tamper_monitor.py --path /path/to/file --stored_hash [HASH]
- Requirements: pip install watchdog twilio
- Configuration: Update EMAIL, SMS, and PUSH settings with your credentials.
"""

import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
import argparse
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client as TwilioClient
import requests

# Configuration (Update these with your credentials)
EMAIL = {
    "sender": "your-email@gmail.com",
    "password": "your-app-password",  # Use App Password for Gmail
    "recipient": "recipient-email@example.com",
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587
}

SMS = {
    "account_sid": "your-twilio-sid",
    "auth_token": "your-twilio-auth-token",
    "twilio_number": "+1234567890",  # Twilio phone number
    "recipient_number": "+0987654321"  # Recipient's phone number
}

PUSH = {
    "api_key": "your-push-api-key",  # Placeholder for push service (e.g., FCM server key)
    "device_token": "your-device-token"  # Target device token for push
}

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
        """Send alerts via email, SMS, and mobile push on tamper detection."""
        timestamp = datetime.now().strftime('%I:%M %p KST, %B %d, %Y')
        message = f"ALERT: Tamper detected in {os.path.basename(self.path)} at {timestamp}!\nStored hash: {self.stored_hash.hex()}\nCurrent hash: {current_hash}"

        # Console Alert
        print(message)

        # Email Alert
        try:
            msg = MIMEText(message)
            msg['Subject'] = f"Tamper Alert - {os.path.basename(self.path)}"
            msg['From'] = EMAIL["sender"]
            msg['To'] = EMAIL["recipient"]
            with smtplib.SMTP(EMAIL["smtp_server"], EMAIL["smtp_port"]) as server:
                server.starttls()
                server.login(EMAIL["sender"], EMAIL["password"])
                server.send_message(msg)
            print("Email alert sent.")
        except Exception as e:
            print(f"Email alert failed: {e}")

        # SMS Alert (Twilio)
        try:
            twilio_client = TwilioClient(SMS["account_sid"], SMS["auth_token"])
            twilio_client.messages.create(
                body=message[:160],  # Truncate to 160 characters for SMS
                from_=SMS["twilio_number"],
                to=SMS["recipient_number"]
            )
            print("SMS alert sent.")
        except Exception as e:
            print(f"SMS alert failed: {e}")

        # Mobile Push Alert (Placeholder API)
        try:
            push_payload = {
                "to": PUSH["device_token"],
                "notification": {
                    "title": f"Tamper Alert - {os.path.basename(self.path)}",
                    "body": message[:100]  # Truncate for push notification
                },
                "priority": "high"
            }
            response = requests.post(
                "https://fcm.googleapis.com/fcm/send",  # Replace with actual push API endpoint
                headers={"Authorization": f"key={PUSH['api_key']}", "Content-Type": "application/json"},
                json=push_payload
            )
            if response.status_code == 200:
                print("Push notification sent.")
            else:
                print(f"Push notification failed: {response.text}")
        except Exception as e:
            print(f"Push notification failed: {e}")

class TamperEventHandler(FileSystemEventHandler):
    def __init__(self, monitor: TamperMonitor):
        self.monitor = monitor

    def on_modified(self, event):
        """Handle file modification events."""
        if event.src_path == self.monitor.path:
            print(f"File modified: {os.path.basename(event.src_path)} at {datetime.now().strftime('%I:%M %p KST')}")
            if not self.monitor.check_integrity():
                print("Tamper alert triggered across all channels!")

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
    parser = argparse.ArgumentParser(description="Real-Time Tamper Detection Monitor with Alerts")
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
