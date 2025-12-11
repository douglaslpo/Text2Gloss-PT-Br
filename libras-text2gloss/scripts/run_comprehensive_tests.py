#!/usr/bin/env python3
"""
Comprehensive test script to validate the system with real cases.
"""

import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mt5_syntax_aware import MT5SyntaxAware, MT5Baseline
from utils.tokenizer_utils import MT5TokenizerWrapper
from utils.syntax_processor import PortugueseSyntaxProcessor
from utils.data_loader import LibrasDataLoader
from utils.metrics import TranslationMetrics

# Real test cases
REAL_TEST_CASES = [
    {
        "pt_br": "O menino está jogando bola no parque",
        "expected": "MENINO JOGAR BOLA PARQUE",
        "category": "ação"
    },
    {
        "pt_br": "Qual é o seu telefone?",
        "expected": "QUAL SEU TELEFONE",
        "category": "pergunta"
    },
    {
        "pt_br": "Eu gosto de estudar português",
        "expected": "EU GOSTAR ESTUDAR PORTUGUES",
        "category": "preferência"
    },
    {
        "pt_br": "A casa é grande e bonita",
        "expected": "CASA GRANDE BONITO",
        "category": "descrição"
    },
    {
        "pt_br": "Meu nome é João",
        "expected": "MEU NOME JOAO",
        "category": "identificação"
    },
    {
        "pt_br": "Hoje faz muito calor",
        "expected": "HOJE MUITO CALOR",
        "category": "clima"
    },
    {
        "pt_br": "O professor ensina matemática",
        "expected": "PROFESSOR ENSINAR MATEMATICA",
        "category": "educação"
    },
    {
        "pt_br": "Eu preciso ir ao médico",
        "expected": "EU PRECISAR IR MEDICO",
        "category": "saúde"
    },
    {
        "pt_br": "A biblioteca fica na rua principal",
        "expected": "BIBLIOTECA FICAR RUA PRINCIPAL",
        "category": "localização"
    },
    {
        "pt_br": "Ela comprou um livro interessante",
        "expected": "ELA COMPRAR LIVRO INTERESSANTE",
        "category": "ação passada"
    }
]

def test_single_case(model, tokenizer, syntax_processor, test_case, use_syntax=False, device='cpu'):
    """Test a single case."""
    source = test_case["pt_br"]
    expected = test_case["expected"]
    
    # Tokenize
    encoding = tokenizer.encode(
        source,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_language_token='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get syntax if needed
    syntax_ids = None
    if use_syntax and syntax_processor:
        try:
            tokens, dep_tags = syntax_processor.process_sentence(source)
            subword_tokens = tokenizer.tokenize(source)
            aligned_tags = syntax_processor.align_with_subwords(tokens, dep_tags, subword_tokens)
            syntax_ids = torch.tensor([aligned_tags[:64] + [38] * (64 - len(aligned_tags[:64]))], 
                                     dtype=torch.long).to(device)
        except Exception as e:
            print(f"  Warning: Syntax processing failed: {e}")
    
    # Generate
    model.eval()
    with torch.no_grad():
        try:
            if use_syntax and syntax_ids is not None:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    syntax_ids=syntax_ids,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True
                )
            else:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=64,
                    num_beams=3,
                    early_stopping=True
                )
            
            pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            return pred_text, expected
        except Exception as e:
            return f"ERROR: {e}", expected

def run_comprehensive_tests():
    """Run comprehensive tests on real cases."""
    print("=" * 80)
    print("Testes Abrangentes com Casos Reais")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}\n")
    
    # Initialize components
    print("Inicializando componentes...")
    tokenizer = MT5TokenizerWrapper(model_name="google/mt5-small")
    syntax_processor = PortugueseSyntaxProcessor()
    metrics_calc = TranslationMetrics()
    
    # Initialize models
    print("Carregando modelos...")
    baseline_model = MT5Baseline(model_name="google/mt5-small")
    syntax_model = MT5SyntaxAware(model_name="google/mt5-small", num_syntax_tags=50)
    
    baseline_model.to(device)
    syntax_model.to(device)
    
    # Test both models
    results = {
        "baseline": {"predictions": [], "references": []},
        "syntax_aware": {"predictions": [], "references": []}
    }
    
    print("\n" + "=" * 80)
    print("Testando Modelo Baseline")
    print("=" * 80)
    
    for i, test_case in enumerate(tqdm(REAL_TEST_CASES, desc="Baseline")):
        pred, ref = test_single_case(
            baseline_model, tokenizer, None, test_case, use_syntax=False, device=device
        )
        results["baseline"]["predictions"].append(pred)
        results["baseline"]["references"].append(ref)
    
    print("\n" + "=" * 80)
    print("Testando Modelo Syntax-Aware")
    print("=" * 80)
    
    for i, test_case in enumerate(tqdm(REAL_TEST_CASES, desc="Syntax-Aware")):
        pred, ref = test_single_case(
            syntax_model, tokenizer, syntax_processor, test_case, use_syntax=True, device=device
        )
        results["syntax_aware"]["predictions"].append(pred)
        results["syntax_aware"]["references"].append(ref)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("Calculando Métricas")
    print("=" * 80)
    
    baseline_metrics = metrics_calc.compute_all_metrics(
        results["baseline"]["predictions"],
        results["baseline"]["references"],
        measure_time=False
    )
    
    syntax_metrics = metrics_calc.compute_all_metrics(
        results["syntax_aware"]["predictions"],
        results["syntax_aware"]["references"],
        measure_time=False
    )
    
    print("\nResultados Baseline:")
    metrics_calc.print_metrics(baseline_metrics)
    
    print("\nResultados Syntax-Aware:")
    metrics_calc.print_metrics(syntax_metrics)
    
    # Save detailed results
    detailed_results = {
        "test_cases": REAL_TEST_CASES,
        "baseline": {
            "predictions": results["baseline"]["predictions"],
            "metrics": baseline_metrics
        },
        "syntax_aware": {
            "predictions": results["syntax_aware"]["predictions"],
            "metrics": syntax_metrics
        }
    }
    
    output_file = Path("results/comprehensive_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Resultados salvos em {output_file}")
    
    # Print examples
    print("\n" + "=" * 80)
    print("Exemplos de Traduções")
    print("=" * 80)
    
    for i in range(min(5, len(REAL_TEST_CASES))):
        print(f"\nExemplo {i+1}: {REAL_TEST_CASES[i]['category']}")
        print(f"  PT-BR: {REAL_TEST_CASES[i]['pt_br']}")
        print(f"  Esperado: {REAL_TEST_CASES[i]['expected']}")
        print(f"  Baseline: {results['baseline']['predictions'][i]}")
        print(f"  Syntax-Aware: {results['syntax_aware']['predictions'][i]}")

if __name__ == "__main__":
    run_comprehensive_tests()

