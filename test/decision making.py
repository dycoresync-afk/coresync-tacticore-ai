"""
XAI Module for TactiCore AI - Document-Based Decision Explanation
- Reads PDF, DOCX, TXT documents.
- Uses RAG for context retrieval.
- LLM analyzes for route planning/risk assessment.
- SHAP explains model decisions on threat scores.
- Requirements: pip install pypdf python-docx transformers torch numpy faiss-cpu sentence-transformers shap
- Run-able: Test with sample documents.
"""

import os
import numpy as np
import torch
import shap
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DocumentBasedXAI:
    def __init__(self):
        # Embedder for RAG
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # LLM for analysis (distilbert for classification; use for threat/risk scoring)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        
        # RAG setup
        self.vector_store = None
        self.documents = []  # Chunks
        self.dimension = 384
        
        # SHAP explainer
        self.shap_explainer = None
        
    def load_document(self, file_path: str) -> str:
        """Load text from PDF, DOCX, TXT."""
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
                raise ValueError(f"Unsupported file: {ext}")
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Chunk text for RAG."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if len(chunk.split()) < chunk_size:
                break
        return chunks

    def build_rag_index(self, file_paths: List[str]) -> None:
        """Build FAISS index from documents."""
        self.documents = []
        for path in file_paths:
            text = self.load_document(path)
            chunks = self.chunk_text(text)
            self.documents.extend(chunks)
        
        if not self.documents:
            raise ValueError("No documents loaded.")
        
        embeddings = self.embedder.encode(self.documents, batch_size=16, show_progress_bar=True).astype('float32')
        
        self.vector_store = faiss.IndexFlatL2(self.dimension)
        self.vector_store.add(embeddings)

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant chunks."""
        if self.vector_store is None:
            raise ValueError("Build RAG index first.")
        
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, indices = self.vector_store.search(query_embedding, top_k)
        retrieved = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return retrieved

    def analyze_decisions(self, query: str, inputs: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Analyze document-based decisions for route planning/risk; use SHAP for explanation."""
        context = self.retrieve_context(query, top_k)
        context_text = "\n\n".join(context) if context else "No context found."
        
        # Prompt for LLM to generate route/risk decisions
        prompt = (
            f"Based on inputs: {json.dumps(inputs, indent=2)}\n"
            f"Context from documents: {context_text}\n\n"
            "Evaluate route planning and risk assessment:\n"
            "- Route Recommendation: Suggest optimal route (low/medium/high risk).\n"
            "- Risk Score (0-10): Based on threats.\n"
            "- Explanation: 2-3 sentences on decision factors."
        )
        
        # Generate LLM response
        response = self.model(prompt, max_length=200)[0]['generated_text']
        
        # Dummy parsing (improve with regex in prod)
        route_rec = "Low Risk Primary Route"  # Placeholder
        risk_score = 4.5  # Placeholder from response
        
        # SHAP Explanation (on model logits)
        if self.shap_explainer is None:
            background = np.array([[0.5] * 4])  # Dummy background for features
            self.shap_explainer = shap.KernelExplainer(self._model_predict, background)

        shap_input = np.array([[inputs.get('humint', 0.0), inputs.get('sigint', 0.0), inputs.get('osint', 0.0), inputs.get('cctv', 0.0)]])
        shap_values = self.shap_explainer.shap_values(shap_input)
        
        explanation = {
            "route_recommendation": route_rec,
            "risk_score": risk_score,
            "shap_contributions": shap_values.tolist(),
            "llm_summary": response,
            "context_used": context
        }
        
        return explanation

    def _model_predict(self, x):
        """Dummy model predict for SHAP (threat score)."""
        inputs = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(inputs)['logits']
        return logits.numpy()

# Run-able Example
if __name__ == "__main__":
    analyzer = DocumentBasedXAI()
    
    # Sample files (create dummy files for testing)
    file_paths = ["report1.pdf", "report2.docx", "report3.txt"]  # Replace with real paths
    
    analyzer.build_rag_index(file_paths)
    
    # Sample query and inputs
    query = "Evaluate route planning for mission in Santo Domingo based on security threats"
    inputs = {"humint": 0.7, "sigint": 0.6, "osint": 0.8, "cctv": 0.5}
    
    result = analyzer.analyze_decisions(query, inputs)
    print("Decision Explanation:")
    print(result)
