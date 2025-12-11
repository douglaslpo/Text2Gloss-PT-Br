#!/usr/bin/env python3
"""
Script master para executar o pipeline completo:
1. Pré-processamento (se necessário)
2. Treinamento Baseline
3. Treinamento Syntax-Aware
4. Avaliação
5. Visualizações
6. Testes completos
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_command(cmd, description, check=True):
    """Executa um comando e exibe progresso."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - start_time
    
    if result.returncode != 0 and check:
        print(f"\n❌ Erro ao executar: {description}")
        sys.exit(1)
    else:
        print(f"\n✅ {description} concluído em {elapsed:.2f}s")
    
    return result.returncode == 0

def main():
    """Executa o pipeline completo."""
    print("\n" + "="*80)
    print("🎯 PIPELINE COMPLETO - PT-BR → LIBRAS Text2Gloss")
    print("="*80)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    venv_python = base_dir / "venv" / "bin" / "python"
    
    # Verificar se venv existe
    if not venv_python.exists():
        print("⚠️  Ambiente virtual não encontrado. Usando Python do sistema.")
        python_cmd = "python3"
    else:
        python_cmd = str(venv_python)
    
    results = {}
    
    # 1. Verificar dados processados
    print("\n📊 Verificando dados processados...")
    processed_dir = base_dir / "data" / "processed"
    if not (processed_dir / "train.json").exists():
        print("⚠️  Dados não processados. Executando pré-processamento...")
        success = run_command(
            [python_cmd, str(scripts_dir / "preprocess_data.py")],
            "Pré-processamento de dados",
            check=False
        )
        results["preprocess"] = success
    else:
        print("✅ Dados já processados.")
        results["preprocess"] = True
    
    # 2. Treinamento Baseline
    print("\n📈 Iniciando treinamento Baseline...")
    baseline_output = base_dir / "models" / "checkpoints" / "baseline_full"
    baseline_output.mkdir(parents=True, exist_ok=True)
    
    success = run_command(
        [python_cmd, str(scripts_dir / "train_mt5_baseline.py"),
         "--config", "configs/training_config.yaml",
         "--output", str(baseline_output)],
        "Treinamento Modelo Baseline",
        check=False
    )
    results["train_baseline"] = success
    
    # 3. Treinamento Syntax-Aware
    print("\n📈 Iniciando treinamento Syntax-Aware...")
    syntax_output = base_dir / "models" / "checkpoints" / "syntax_aware_full"
    syntax_output.mkdir(parents=True, exist_ok=True)
    
    success = run_command(
        [python_cmd, str(scripts_dir / "train_mt5_syntax.py"),
         "--config", "configs/training_config.yaml",
         "--output", str(syntax_output)],
        "Treinamento Modelo Syntax-Aware",
        check=False
    )
    results["train_syntax"] = success
    
    # 4. Avaliação
    print("\n📊 Iniciando avaliação...")
    baseline_model = baseline_output / "best_model"
    syntax_model = syntax_output / "best_model"
    
    # Se não houver best_model, usar o último checkpoint
    if not baseline_model.exists():
        checkpoints = list(baseline_output.glob("checkpoint-*"))
        if checkpoints:
            baseline_model = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    
    if not syntax_model.exists():
        checkpoints = list(syntax_output.glob("checkpoint-*"))
        if checkpoints:
            syntax_model = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    
    if baseline_model.exists() and syntax_model.exists():
        success = run_command(
            [python_cmd, str(scripts_dir / "evaluate.py"),
             "--model_baseline", str(baseline_model),
             "--model_syntax", str(syntax_model),
             "--test_data", str(processed_dir / "test.json")],
            "Avaliação de Modelos",
            check=False
        )
        results["evaluate"] = success
    else:
        print("⚠️  Modelos não encontrados. Pulando avaliação.")
        results["evaluate"] = False
    
    # 5. Testes Completos
    print("\n🧪 Executando testes completos...")
    success = run_command(
        [python_cmd, str(scripts_dir / "run_comprehensive_tests.py")],
        "Testes Completos",
        check=False
    )
    results["comprehensive_tests"] = success
    
    # 6. Visualizações
    print("\n📊 Gerando visualizações...")
    success = run_command(
        [python_cmd, str(scripts_dir / "generate_visualizations.py")],
        "Geração de Visualizações",
        check=False
    )
    results["visualizations"] = success
    
    # Resumo final
    print("\n" + "="*80)
    print("📋 RESUMO DO PIPELINE")
    print("="*80)
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step.replace('_', ' ').title()}")
    print("="*80)
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Verificar se tudo foi bem-sucedido
    all_success = all(results.values())
    if all_success:
        print("🎉 Pipeline completo executado com sucesso!")
    else:
        print("⚠️  Algumas etapas falharam. Verifique os logs acima.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

