"""
Tamper-Proof Log Tracker on Solana for TactiCore AI
- Logs mission events (e.g., route/risk decisions) as immutable on-chain memos.
- Integrates with XAI for auditable insights.
- Alpha stage: Uses Solana Note program for simple, verifiable logs.
"""

from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.transaction import Transaction
from solana.system_program import transfer, TransferParams
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.note_program import create_memo, create_memo_account
import json
import base64
from typing import Dict, Any
from datetime import datetime

# Configuration
RPC_URL = "https://api.devnet.solana.com"  # Switch to mainnet-beta for production
NOTE_PROGRAM_ID = PublicKey("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")  # Solana Note Program

class SolanaLogTracker:
    def __init__(self, payer_keypair_path: str = None):
        """
        Initialize with payer keypair (JSON file from solana-keygen).
        Args:
            payer_keypair_path: Path to payer keypair JSON.
        """
        self.client = Client(RPC_URL)
        if payer_keypair_path:
            with open(payer_keypair_path, 'r') as f:
                self.payer = Keypair.from_secret_key(bytes(json.load(f)))
        else:
            self.payer = Keypair.generate()  # Dev keypair - replace for prod
        print(f"Initialized with payer: {self.payer.public_key}")

    def _serialize_event(self, event: Dict[str, Any]) -> str:
        """Serialize event to JSON for on-chain memo."""
        event["timestamp"] = datetime.utcnow().isoformat()
        return json.dumps(event)

    def log_event(self, event_type: str, details: Dict[str, Any], lamports: int = 1000) -> str:
        """
        Log event to Solana as a tamper-proof transaction.
        Args:
            event_type: e.g., "route_decision" or "risk_assessment".
            details: Dict with XAI outputs (e.g., {"route_scores": [0.3, 0.7], "risk_score": 0.45}).
            lamports: Micro-lamports for tx fee (minimal for log).
        Returns:
            Tx signature (immutable log reference).
        """
        try:
            # Serialize
            memo_text = self._serialize_event({"type": event_type, "details": details})
            
            # Create memo instruction
            memo_ix = create_memo(memo_text.encode())
            
            # Build transaction with small transfer for fee coverage
            tx = Transaction().add(memo_ix)
            tx.add(transfer(TransferParams(
                from_pubkey=self.payer.public_key,
                to_pubkey=self.payer.public_key,  # Self-transfer for minimal tx
                lamports=lamports
            )))

            # Sign and send
            tx.sign(self.payer)
            result = self.client.send_transaction(tx, opts=TxOpts(skip_preflight=True))
            
            print(f"Logged '{event_type}' on Solana: {result.value}")
            return result.value  # Tx signature for audit

        except Exception as e:
            print(f"Log failed: {e}")
            return None

    def query_log(self, tx_signature: str) -> Dict[str, Any]:
        """Query a log transaction for verification."""
        try:
            sig = PublicKey(tx_signature)
            tx_details = self.client.get_transaction(sig, encoding="jsonParsed")
            if tx_details.value:
                # Extract memo from meta (logs or instructions)
                logs = tx_details.value.transaction.meta.log_messages
                memo_log = [log for log in logs if "Program log: Memo" in log]
                if memo_log:
                    memo_content = memo_log[0].split("Memo: ")[-1]  # Parse memo
                    return json.loads(memo_content)
            return {"error": "Log not found"}
        except Exception as e:
            print(f"Query failed: {e}")
            return {"error": str(e)}

# Example usage (integrate with XAI)
if __name__ == "__main__":
    tracker = SolanaLogTracker()  # Add keypair path for prod
    
    # Sample XAI output for route/risk
    xai_output = {
        "route_scores": [0.3, 0.7, 0.0],  # A: Safe, B: Fast, C: Avoid
        "risk_score": 0.45,
        "explanation": "OSINT (0.8) boosts Route B score; HUMINT (0.7) raises risk."
    }
    
    # Log route decision
    tx_sig = tracker.log_event("route_decision", xai_output)
    print(f"Tx Signature: {tx_sig}")
    
    # Query for tamper-proof verification
    if tx_sig:
        log = tracker.query_log(tx_sig)
        print("Retrieved Log:", log)
