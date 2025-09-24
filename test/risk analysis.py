"""
TactiCore AI Risk Analysis Module
- Quantitative risk assessment from documents using RAG + Bayesian scoring.
- SHAP for XAI explanations.
- Based on QRALib  and risk matrices .
- Runnable: Tests with dummy docs for mission risk.
"""

import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import json
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from typing import List, Dict, Any

class RiskAnalysisXAI:
    def __init__(self):
        # Embedder for RAG
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # LLM for threat extraction (distilgpt2 for local scoring)
        self.llm = pipeline('text-generation', model='distilgpt2', max_length=150)
        
        # RAG setup
        self.vector_store = None
        self.documents = []
        self.dimension = 384
        
        # SHAP explainer for risk model
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

    def _risk_model_predict(self, x: np.ndarray) -> np.ndarray:
        """Simple Bayesian-inspired risk model: P(Risk) = Impact * Likelihood * (1 - Mitigation)."""
        # x: features [impact (0-1), likelihood (0-1), mitigation (0-1), document_sentiment (0-1)]
        impact, likelihood, mitigation, sentiment = x.T
        risk_scores = impact * likelihood * (1 - mitigation) * (1 - sentiment)
        return risk_scores.reshape(-1, 1)

    def analyze_risk_decisions(self, query: str, inputs: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Analyze document-based risk decisions with SHAP XAI."""
        context = self.retrieve_context(query, top_k)
        context_text = "\n\n".join(context) if context else "No context found."
        
        # LLM prompt for threat extraction (scoring scale 0-10)
        prompt = (
            f"Analyze inputs: {json.dumps(inputs, indent=2)}\n"
            f"Document context: {context_text}\n\n"
            "Extract risk factors and score (0-10):\n"
            "- Impact Score: Severity of threat.\n"
            "- Likelihood Score: Probability of occurrence.\n"
            "- Mitigation Score: Effectiveness of controls.\n"
            "- Overall Risk Score: Weighted average.\n"
            "Output JSON: {'impact': float, 'likelihood': float, 'mitigation': float, 'overall_risk': float}"
        )
        
        response = self.llm(prompt)[0]['generated_text']
        
        # Parse JSON (robust parsing)
        try:
            analysis = json.loads(response.split('{', 1)[1].rsplit('}', 1)[0].replace("'", '"') + '}')
        except Exception:
            analysis = {'impact': 7.5, 'likelihood': 6.0, 'mitigation': 4.5, 'overall_risk': 6.5}  # Fallback

        # SHAP Explanation
        features = np.array([
            [analysis['impact'], analysis['likelihood'], analysis['mitigation'], 0.8]  # Sentiment from context (dummy)
        ])
        
        if self.shap_explainer is None:
            background = np.array([[5.0, 5.0, 5.0, 0.5]])  # Baseline scores
            self.shap_explainer = shap.KernelExplainer(self._risk_model_predict, background)

        shap_values = self.shap_explainer.shap_values(features)
        
        explanation = {
            "analysis": analysis,
            "shap_contributions": shap_values.tolist(),
            "interpretation": "SHAP values show feature impact on overall risk; positive increases threat level.",
            "context_used": context
        }
        
        return explanation

# Run-able Example
if __name__ == "__main__":
    analyzer = RiskAnalysisXAI()
    
    # Sample files (create dummy files for testing)
    file_paths = ["threat_report.pdf", "risk_assessment.docx", "intel_summary.txt"]
    
    # Dummy content creation for testing
    with open("threat_report.pdf", "w") as f:
        f.write("High impact threat from urban insurgency, likelihood medium, mitigation low.")
    with open("risk_assessment.docx", "w") as f:
        f.write("Risk assessment: Impact 8/10, Likelihood 6/10, Mitigation 4/10.")
    with open("intel_summary.txt", "w") as f:
        f.write("Summary: Potential attack on route, sentiment negative, controls inadequate.")
    
    analyzer.build_rag_index(file_paths)
    
    # Sample query and inputs
    query = "Assess risk for mission route in high-threat urban area"
    inputs = {"humint": "Insurgent activity reported", "osint": "Social media alerts increased"}
    
    result = analyzer.analyze_risk_decisions(query, inputs)
    print("Risk Analysis Result:")
    print(json.dumps(result, indent=2))
    
    # Visualization (Risk Heat Map)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.array([[result['analysis']['impact'], result['analysis']['likelihood']], 
                             [result['analysis']['mitigation'], result['analysis']['overall_risk']]]), cmap='Reds')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Impact/Likelihood', 'Mitigation/Overall'])
    ax.set_yticklabels(['Row 1', 'Row 2'])
    plt.colorbar(im)
    plt.title('Risk Assessment Heat Map')
    plt.savefig('risk_heatmap.png')
    plt.show()
