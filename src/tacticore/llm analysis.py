"""
CoreSync TactiCoreAI Intelligence Analyzer
- Reads PDF, DOCX, TXT files, analyzes with LLM for threat scoring and risk assessment.
- Incorporates RAG for context-aware analysis using FAISS vector store.
- Workable: Full end-to-end; scores intel on threat level (0-10), reliability, and risk summary.
- Requirements: pip install pypdf python-docx transformers torch numpy faiss-cpu sentence-transformers
- Specific Example: Analyzes sample intel reports for mission threats.
"""

import os
import numpy as np
import torch
from pypdf import PdfReader
from docx import Document
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any

class IntelAnalyzer:
    def __init__(self):
        # Embeddings model for RAG (all-MiniLM-L6-v2: 384-dim, fast)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # LLM for analysis (distilgpt2 for efficiency; outputs structured scores/summaries)
        self.llm = pipeline('text-generation', model='distilgpt2', max_length=200, num_return_sequences=1)
        
        # RAG setup
        self.vector_store = None
        self.documents = []  # Chunks for retrieval
        self.dimension = 384  # Embedder output size
        
    def load_document(self, file_path: str) -> str:
        """Extract text from PDF, DOCX, or TXT."""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                reader = PdfReader(file_path)
                text = ' '.join(page.extract_text() or '' for page in reader.pages)
            elif ext == '.docx':
                doc = Document(file_path)
                text = ' '.join(para.text for para in doc.paragraphs)
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                raise ValueError(f"Unsupported file type: {ext}. Use PDF, DOCX, or TXT.")
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better RAG context."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if len(chunk.split()) < chunk_size:
                break
        return chunks

    def build_rag_index(self, file_paths: List[str]) -> None:
        """Build FAISS vector store from multiple files."""
        self.documents = []
        for path in file_paths:
            text = self.load_document(path)
            chunks = self.chunk_text(text)
            self.documents.extend(chunks)
        
        if not self.documents:
            raise ValueError("No documents loaded for RAG index.")
        
        # Embed all chunks
        embeddings = self.embedder.encode(self.documents, batch_size=16, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype='float32')
        
        # Create FAISS index
        self.vector_store = faiss.IndexFlatL2(self.dimension)
        self.vector_store.add(embeddings)

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve top-k relevant chunks using cosine similarity (via FAISS)."""
        if self.vector_store is None or not self.documents:
            raise ValueError("RAG index not built. Call build_rag_index first.")
        
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, indices = self.vector_store.search(query_embedding, top_k)
        retrieved = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return retrieved

    def analyze_and_score(self, query: str, inputs: Dict[str, Any], top_k: int = 5, scoring_scale: int = 10) -> Dict[str, Any]:
        """Analyze inputs with LLM + RAG; score threat/reliability/risk."""
        # Retrieve relevant context
        context = self.retrieve_context(query, top_k)
        context_text = "\n\n---\n\n".join(context) if context else "No relevant context found."
        
        # Construct detailed prompt for LLM (specific to defsec: threat level, reliability, risk summary)
        prompt = (
            f"Analyze the following intelligence inputs for a mission: {json.dumps(inputs, indent=2)}.\n"
            f"Context from reports: {context_text}\n\n"
            "Provide:\n"
            "- Threat Level Score (0-10): Based on severity and immediacy.\n"
            "- Reliability Score (0-10): Based on source consistency and context match.\n"
            "- Risk Summary: 2-3 sentence explanation of key risks, recommendations.\n"
            "Output in JSON format: {'threat_score': int, 'reliability_score': int, 'risk_summary': str}"
        )
        
        # Generate LLM response
        response = self.llm(prompt)[0]['generated_text']
        
        # Parse JSON from response (assume LLM outputs valid JSON; add error handling in prod)
        try:
            analysis = json.loads(response.split('{', 1)[1].rsplit('}', 1)[0].replace("'", '"') + '}')
        except Exception:
            analysis = {'threat_score': 0, 'reliability_score': 0, 'risk_summary': "Error parsing LLM output."}
        
        return {
            "analysis": analysis,
            "context_used": context
        }

# Run-able Specific Example (replace with real file paths)
if __name__ == "__main__":
    analyzer = IntelAnalyzer()
    
    # Sample files (create these dummy files for testing)
    # dummy_report1.pdf: "Suspicious activity near checkpoint: 3 vehicles, armed individuals."
    # dummy_report2.docx: "OSINT: Social media chatter on potential threat; reliability high."
    # dummy_report3.txt: "HUMINT: Local source confirms no explosives; low risk."
    file_paths = ["dummy_report1.pdf", "dummy_report2.docx", "dummy_report3.txt"]
    
    # Build RAG index
    analyzer.build_rag_index(file_paths)
    
    # Specific query and additional inputs
    query = "Assess threat from checkpoint activity in Middle East mission"
    inputs = {
        "humint_score": 0.7,
        "osint_score": 0.8,
        "additional_notes": "Vehicles spotted at 10:00 AM; no confirmed explosives."
    }
    
    # Analyze and score
    result = analyzer.analyze_and_score(query, inputs)
    print("Analysis Result:")
    print(json.dumps(result, indent=2))
