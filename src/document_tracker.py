"""
Solana Document Tracking for DoD Cybersecurity Compliance
- Hashes documents (PDF/DOCX/TXT) and logs on-chain for tamper-proof audit.
- Verifies integrity via blockchain query.
- Aligns with NIST 800-171/CMMC: Immutable logs, chain of custody.
- Runnable on Devnet; dummy files generated for testing.
"""

import os
import hashlib
import json
from datetime import datetime
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.transaction import Transaction
from solana.system_program import transfer, TransferParams
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.note_program import create_memo
from pypdf import PdfReader
from docx import Document
from typing import Dict, Any, List

# Configuration (Devnet for testing; switch to mainnet for prod)
RPC_URL = "https://api.devnet.solana.com"
NOTE_PROGRAM_ID = PublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")  # Solana Note Program
LAMPORTS_FEE = 1000  # Minimal tx fee

class DocumentTracker:
    def __init__(self, payer_keypair_path: str = None):
        """Initialize with payer keypair (generate with `solana-keygen new`)."""
        self.client = Client(RPC_URL)
        if payer_keypair_path:
            with open(payer_keypair_path, 'r') as f:
                self.payer = Keypair.from_secret_key(bytes(json.load(f)))
        else:
            # Generate dev keypair for testing (insecure for prod!)
            self.payer = Keypair.generate()
            print(f"Generated dev keypair: {self.payer.public_key}")
        print(f"Initialized tracker with payer: {self.payer.public_key}")

    def load_document_hash(self, file_path: str) -> bytes:
        """Load and hash document (SHA-256 for DoD integrity)."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                reader = PdfReader(file_path)
                text = ''.join(page.extract_text() or '' for page in reader.pages)
            elif ext == '.docx':
                doc = Document(file_path)
                text = '\n'.join(para.text for para in doc.paragraphs)
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                raise ValueError(f"Unsupported file: {ext}")
            
            # SHA-256 hash for immutability
            hash_object = hashlib.sha256(text.encode('utf-8'))
            return hash_object.digest()
        except Exception as e:
            raise RuntimeError(f"Error hashing {file_path}: {e}")

    def _serialize_event(self, event_type: str, doc_hash: bytes, metadata: Dict[str, Any]) -> str:
        """Serialize event for on-chain memo."""
        metadata['type'] = event_type
        metadata['doc_hash'] = doc_hash.hex()
        metadata['timestamp'] = datetime.utcnow().isoformat()
        return json.dumps(metadata)

    def track_document(self, file_path: str, event_type: str = "document_upload", metadata: Dict[str, Any] = None) -> str:
        """
        Track document: Hash, log on-chain as memo tx.
        Args:
            file_path: Path to PDF/DOCX/TXT.
            event_type: e.g., "document_upload", "risk_assessment".
            metadata: Dict with details (e.g., {'mission_id': 'SDQ-2026-001', 'user': 'DY'}).
        Returns:
            Tx signature for audit (DoD chain of custody).
        """
        if not metadata:
            metadata = {}
        
        try:
            doc_hash = self.load_document_hash(file_path)
            memo_text = self._serialize_event(event_type, doc_hash, metadata)
            
            # Create memo instruction
            memo_ix = create_memo(memo_text.encode())
            
            # Build tx with minimal self-transfer for fee
            tx = Transaction().add(memo_ix)
            tx.add(transfer(TransferParams(
                from_pubkey=self.payer.public_key,
                to_pubkey=self.payer.public_key,
                lamports=LAMPORTS_FEE
            )))

            # Sign and send
            tx.sign(self.payer)
            result = self.client.send_transaction(tx, opts=TxOpts(skip_preflight=True))
            
            print(f"Document '{os.path.basename(file_path)}' tracked on Solana: {result.value}")
            return result.value  # Tx sig for verification

        except Exception as e:
            print(f"Tracking failed: {e}")
            return None

    def verify_document_integrity(self, file_path: str, tx_signature: str) -> Dict[str, Any]:
        """Verify document against on-chain log (DoD integrity check)."""
        try:
            current_hash = self.load_document_hash(file_path)
            
            # Query tx
            sig = PublicKey(tx_signature)
            tx_details = self.client.get_transaction(sig, encoding="jsonParsed")
            if not tx_details.value:
                return {"status": "error", "message": "Transaction not found"}
            
            # Extract memo from logs
            logs = tx_details.value.transaction.meta.log_messages
            memo_log = next((log for log in logs if "Program log: Memo" in log), None)
            if not memo_log:
                return {"status": "error", "message": "Memo not found in transaction"}
            
            # Parse memo content
            memo_content = memo_log.split("Memo: ")[-1]
            stored_data = json.loads(memo_content)
            stored_hash = bytes.fromhex(stored_data['doc_hash'])
            
            is_valid = current_hash == stored_hash
            return {
                "status": "valid" if is_valid else "tampered",
                "current_hash": current_hash.hex(),
                "stored_hash": stored_hash.hex(),
                "timestamp": stored_data['timestamp'],
                "metadata": {k: v for k, v in stored_data.items() if k != 'doc_hash' and k != 'timestamp'}
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Runnable Example (Standalone for GitHub)
if __name__ == "__main__":
    tracker = DocumentTracker()  # Use dev keypair
    
    # Create dummy document for testing
    dummy_content = "Sample UN mission report: Threat level medium, route via Autopista, risk factors: traffic."
    with open("dummy_report.txt", "w") as f:
        f.write(dummy_content)
    
    # Track document
    tx_sig = tracker.track_document("dummy_report.txt", "mission_report", {"mission_id": "SDQ-2026-001", "user": "DY"})
    print(f"Tx Signature: {tx_sig}")
    
    if tx_sig:
        # Verify integrity
        verification = tracker.verify_document_integrity("dummy_report.txt", tx_sig)
        print("Verification:", verification)
    
    # Clean up dummy file
    os.remove("dummy_report.txt")
