"""
Test Dataset for PharmaBula Evaluation

Provides sample queries for benchmarking RAG implementations.
"""

import json
from pathlib import Path
from typing import Optional


# ============================================================
# Sample Test Queries for Drug Information
# ============================================================

SAMPLE_QUERIES = [
    # Simple medication questions
    {
        "id": "q001",
        "query": "Para que serve o paracetamol?",
        "category": "indicacoes",
        "complexity": "simple",
        "expected_topics": ["dor", "febre", "analgésico"]
    },
    {
        "id": "q002",
        "query": "Qual a dosagem recomendada de ibuprofeno para adultos?",
        "category": "posologia",
        "complexity": "simple",
        "expected_topics": ["mg", "horas", "máximo"]
    },
    {
        "id": "q003",
        "query": "Quais são os efeitos colaterais da dipirona?",
        "category": "efeitos_colaterais",
        "complexity": "simple",
        "expected_topics": ["reações", "alérgicas", "hipotensão"]
    },
    
    # Moderate complexity
    {
        "id": "q004",
        "query": "O omeprazol pode ser tomado junto com o paracetamol?",
        "category": "interacoes",
        "complexity": "moderate",
        "expected_topics": ["interação", "seguro", "recomendação"]
    },
    {
        "id": "q005",
        "query": "Quais as contraindicações do uso de aspirina em gestantes?",
        "category": "contraindicacoes",
        "complexity": "moderate",
        "expected_topics": ["gravidez", "risco", "evitar"]
    },
    {
        "id": "q006",
        "query": "Como funciona o mecanismo de ação do losartana?",
        "category": "mecanismo",
        "complexity": "moderate",
        "expected_topics": ["angiotensina", "receptor", "pressão"]
    },
    {
        "id": "q007",
        "query": "Posso tomar amoxicilina se tenho alergia a penicilina?",
        "category": "contraindicacoes",
        "complexity": "moderate",
        "expected_topics": ["alergia", "reação cruzada", "alternativa"]
    },
    
    # Complex queries
    {
        "id": "q008",
        "query": "Quais são as interações medicamentosas entre varfarina, AAS e omeprazol?",
        "category": "interacoes",
        "complexity": "complex",
        "expected_topics": ["sangramento", "INR", "monitorar"]
    },
    {
        "id": "q009",
        "query": "Explique as diferenças entre os anti-inflamatórios não esteroides (AINEs) e quando cada um é mais indicado",
        "category": "comparacao",
        "complexity": "complex",
        "expected_topics": ["seletivo", "COX-2", "cardiovascular", "gastrointestinal"]
    },
    {
        "id": "q010",
        "query": "Como ajustar a dose de metformina em pacientes com insuficiência renal?",
        "category": "posologia",
        "complexity": "complex",
        "expected_topics": ["clearance", "creatinina", "redução", "contraindicado"]
    },
    
    # Emergency/Safety queries
    {
        "id": "q011",
        "query": "Tomei 3 comprimidos de paracetamol 750mg de uma vez, o que fazer?",
        "category": "emergencia",
        "complexity": "urgent",
        "expected_topics": ["médico", "emergência", "hepatotoxicidade"]
    },
    {
        "id": "q012",
        "query": "Quais os sintomas de intoxicação por lítio?",
        "category": "emergencia",
        "complexity": "urgent",
        "expected_topics": ["tremor", "confusão", "vômito", "hospital"]
    },
    
    # Professional mode queries
    {
        "id": "q013",
        "query": "Qual o perfil farmacocinético do propranolol e suas implicações clínicas?",
        "category": "farmacologia",
        "complexity": "professional",
        "expected_topics": ["biodisponibilidade", "meia-vida", "metabolismo", "CYP"]
    },
    {
        "id": "q014",
        "query": "Descreva o protocolo de dessensibilização para pacientes alérgicos a sulfas que necessitam de tratamento",
        "category": "protocolo",
        "complexity": "professional",
        "expected_topics": ["progressivo", "hospitalar", "monitorização"]
    },
    
    # Edge cases
    {
        "id": "q015",
        "query": "Medicamentos para dor",
        "category": "vago",
        "complexity": "simple",
        "expected_topics": ["analgésicos", "tipos", "indicações"]
    },
    {
        "id": "q016",
        "query": "remédio bom pra gripe",
        "category": "coloquial",
        "complexity": "simple",
        "expected_topics": ["sintomas", "paracetamol", "descongestionante"]
    },
    {
        "id": "q017",
        "query": "Existe alguma alternativa natural ao rivotril?",
        "category": "alternativas",
        "complexity": "moderate",
        "expected_topics": ["ansiedade", "consulta médica", "não substituir"]
    },
    
    # Specific scenarios
    {
        "id": "q018",
        "query": "Minha mãe de 75 anos pode tomar qualquer anti-inflamatório?",
        "category": "idosos",
        "complexity": "moderate",
        "expected_topics": ["renal", "cardiovascular", "gástrico", "cuidados"]
    },
    {
        "id": "q019",
        "query": "Quais medicamentos evitar durante a amamentação?",
        "category": "lactacao",
        "complexity": "moderate",
        "expected_topics": ["contraindicados", "seguro", "consultar"]
    },
    {
        "id": "q020",
        "query": "Posso dirigir depois de tomar clonazepam?",
        "category": "advertencias",
        "complexity": "simple",
        "expected_topics": ["sonolência", "evitar", "risco"]
    }
]


class TestDataset:
    """
    Test dataset manager for evaluation.
    
    Features:
    - Sample query collection
    - Category filtering
    - Complexity filtering
    - Export/import from JSON
    """
    
    def __init__(self, queries: list[dict] = None):
        """Initialize with queries (default to SAMPLE_QUERIES)."""
        self.queries = queries or SAMPLE_QUERIES.copy()
    
    def get_all(self) -> list[dict]:
        """Get all queries."""
        return self.queries
    
    def get_by_category(self, category: str) -> list[dict]:
        """Get queries by category."""
        return [q for q in self.queries if q.get("category") == category]
    
    def get_by_complexity(self, complexity: str) -> list[dict]:
        """Get queries by complexity level."""
        return [q for q in self.queries if q.get("complexity") == complexity]
    
    def get_simple(self) -> list[dict]:
        """Get simple queries (for quick testing)."""
        return self.get_by_complexity("simple")
    
    def get_sample(self, n: int = 5) -> list[dict]:
        """Get a random sample of n queries."""
        import random
        return random.sample(self.queries, min(n, len(self.queries)))
    
    def add_query(self, query: dict):
        """Add a new query to the dataset."""
        self.queries.append(query)
    
    def save(self, filepath: str):
        """Save dataset to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.queries, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "TestDataset":
        """Load dataset from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            queries = json.load(f)
        return cls(queries)
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)


# ============================================================
# Dataset Statistics
# ============================================================

def get_dataset_stats(dataset: TestDataset = None) -> dict:
    """Get statistics about the dataset."""
    dataset = dataset or TestDataset()
    
    from collections import Counter
    
    categories = Counter(q.get("category") for q in dataset)
    complexities = Counter(q.get("complexity") for q in dataset)
    
    return {
        "total_queries": len(dataset),
        "categories": dict(categories),
        "complexities": dict(complexities),
        "avg_query_length": sum(len(q["query"]) for q in dataset) / len(dataset)
    }


# ============================================================
# Save default dataset
# ============================================================

def create_default_dataset(output_path: str = None):
    """Create and save the default test dataset."""
    output_path = output_path or "./data/test_queries.json"
    dataset = TestDataset()
    dataset.save(output_path)
    return output_path
