import os
import re
import sys
import pandas as pd
from datetime import datetime

# Configuração dos diretórios
CROPS_DIR = "crops"
WEAPON_DETECTIONS_DIR = "weapon_detections"

# Limiares de confiança
CONFIDENCE_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Limiares de fração para análise por sample
FRACTION_THRESHOLDS = [0.1, 0.25, 0.5, 0.75, 0.9]

# Modo de contagem de frames
# True: Contar frames únicos (cada frame conta uma vez - Opção A)
# False: Contar classificações (frames podem contar múltiplas vezes - Opção B)
COUNT_UNIQUE_FRAMES = True

# Crops anotados manualmente como falsos positivos
MANUAL_FALSE_POSITIVES = [
    "real_10_02_clip_001_1080p_compressed_10fps_every10frames/frame_0004_person_02_weapon_01_gun_conf_0.14.jpg",
    "real_10_02_clip_001_1080p_compressed_10fps_every10frames/frame_0002_person_02_weapon_01_gun_conf_0.18.jpg",
    "real_05_05_clip_007_1080p_compressed_10fps_every10frames/frame_0004_person_01_weapon_03_gun_conf_0.10.jpg",
    "real_05_05_clip_001_1080p_compressed_10fps_every10frames/frame_0002_person_01_weapon_02_gun_conf_0.13.jpg",
    "real_05_02_clip_007_1080p_compressed_10fps_every10frames/frame_0000_person_01_weapon_01_gun_conf_0.10.jpg",
    "real_05_02_clip_007_1080p_compressed_10fps_every10frames/frame_0004_person_01_weapon_03_gun_conf_0.11.jpg",
    "real_05_02_clip_007_1080p_compressed_10fps_every10frames/frame_0002_person_01_weapon_02_gun_conf_0.13.jpg",
    "real_05_02_clip_001_1080p_compressed_10fps_every10frames/frame_0002_person_01_weapon_01_gun_conf_0.31.jpg",
    "real_05_02_clip_000_1080p_compressed_10fps_every10frames/frame_0008_person_01_weapon_06_gun_conf_0.13.jpg",
    "real_05_02_clip_000_1080p_compressed_10fps_every10frames/frame_0005_person_01_weapon_04_gun_conf_0.19.jpg",
    "real_05_02_clip_000_1080p_compressed_10fps_every10frames/frame_0004_person_01_weapon_04_gun_conf_0.13.jpg",
    "real_05_02_clip_000_1080p_compressed_10fps_every10frames/frame_0002_person_01_weapon_05_gun_conf_0.19.jpg",
]

# Número de detecções de pessoa por sample
PERSON_DETECTIONS_PER_SAMPLE = {
    "falso_05_02_clip_000_1080p_compressed_10fps_every10frames": 10,
    "falso_05_02_clip_001_1080p_compressed_10fps_every10frames": 10,
    "falso_05_02_clip_007_1080p_compressed_10fps_every10frames": 10,
    "falso_05_05_clip_000_1080p_compressed_10fps_every10frames": 10,
    "falso_05_05_clip_001_1080p_compressed_10fps_every10frames": 10,
    "falso_05_05_clip_007_1080p_compressed_10fps_every10frames": 9,
    "falso_10_02_clip_000_1080p_compressed_10fps_every10frames": 10,
    "falso_10_02_clip_001_1080p_compressed_10fps_every10frames": 10,
    "falso_10_02_clip_007_1080p_compressed_10fps_every10frames": 10,
    "falso_10_05_clip_000_1080p_compressed_10fps_every10frames": 10,
    "falso_10_05_clip_001_1080p_compressed_10fps_every10frames": 10,
    "falso_10_05_clip_007_1080p_compressed_10fps_every10frames": 10,
    "real_05_02_clip_000_1080p_compressed_10fps_every10frames": 10,
    "real_05_02_clip_001_1080p_compressed_10fps_every10frames": 10,
    "real_05_02_clip_007_1080p_compressed_10fps_every10frames": 10,
    "real_05_05_clip_000_1080p_compressed_10fps_every10frames": 10,
    "real_05_05_clip_001_1080p_compressed_10fps_every10frames": 10,
    "real_05_05_clip_007_1080p_compressed_10fps_every10frames": 10,
    "real_10_02_clip_000_1080p_compressed_10fps_every10frames": 10,
    "real_10_02_clip_001_1080p_compressed_10fps_every10frames": 12,
    "real_10_02_clip_007_1080p_compressed_10fps_every10frames": 10,
    "real_10_05_clip_000_1080p_compressed_10fps_every10frames": 8,
    "real_10_05_clip_001_1080p_compressed_10fps_every10frames": 10,
    "real_10_05_clip_007_1080p_compressed_10fps_every10frames": 9,
}


def extract_confidence_from_filename(filename):
    """Extrai a confiança do nome do arquivo."""
    match = re.search(r'_gun_conf_(\d+\.\d+)\.jpg', filename)
    if match:
        return float(match.group(1))
    return None


def extract_frame_from_filename(filename):
    """Extrai o número do frame do nome do arquivo."""
    match = re.search(r'frame_(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def extract_person_from_filename(filename):
    """Extrai o ID da pessoa do nome do arquivo."""
    match = re.search(r'person_(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def get_frames_in_crops(crops_dir, sample_pattern):
    """Retorna um conjunto de crops (sample, frame, person) que existem."""
    crops = set()
    for sample in os.listdir(crops_dir):
        if sample_pattern in sample:
            sample_path = os.path.join(crops_dir, sample)
            if os.path.isdir(sample_path):
                for crop_file in os.listdir(sample_path):
                    frame_num = extract_frame_from_filename(crop_file)
                    person_id = extract_person_from_filename(crop_file)
                    if frame_num and person_id:
                        crops.add((sample, frame_num, person_id))
    return crops


def get_weapon_detections_by_confidence(weapon_dir, min_confidence):
    """Retorna detecções de arma agrupadas por sample, frame e person."""
    detections = {}
    
    if not os.path.exists(weapon_dir):
        return detections
    
    for sample in os.listdir(weapon_dir):
        sample_path = os.path.join(weapon_dir, sample)
        if os.path.isdir(sample_path):
            detections[sample] = {}
            
            for weapon_file in os.listdir(sample_path):
                confidence = extract_confidence_from_filename(weapon_file)
                frame_num = extract_frame_from_filename(weapon_file)
                person_id = extract_person_from_filename(weapon_file)
                
                if confidence is not None and frame_num is not None and person_id is not None:
                    if confidence >= min_confidence:
                        key = (frame_num, person_id)
                        if key not in detections[sample]:
                            detections[sample][key] = []
                        
                        full_path = f"{sample}/{weapon_file}"
                        detections[sample][key].append({
                            'file': weapon_file,
                            'full_path': full_path,
                            'confidence': confidence
                        })
    
    return detections


def analyze_by_frame(crops_dir, weapon_dir, confidence_threshold):
    """Analisa métricas por crop (detecção de pessoa)."""
    # Obter todos os crops de pessoa
    crops_falso = get_frames_in_crops(crops_dir, "falso")
    crops_real = get_frames_in_crops(crops_dir, "real")
    
    # Obter detecções de arma
    weapon_detections = get_weapon_detections_by_confidence(weapon_dir, confidence_threshold)
    
    # Criar conjunto de crops com detecções de arma
    weapon_crops = set()
    weapon_crops_with_manual_fp = {}  # crops que têm detecções anotadas manualmente
    
    for sample, detections in weapon_detections.items():
        for (frame_num, person_id), crops in detections.items():
            crop_key = (sample, frame_num, person_id)
            weapon_crops.add(crop_key)
            
            # Verificar se alguma detecção é um falso positivo manual
            manual_fps = []
            other_detections = []
            
            for detection in crops:
                if detection['full_path'] in MANUAL_FALSE_POSITIVES:
                    manual_fps.append(detection['full_path'])
                else:
                    other_detections.append(detection['full_path'])
            
            if manual_fps:
                weapon_crops_with_manual_fp[crop_key] = {
                    'manual_fps': manual_fps,
                    'other_detections': other_detections
                }
    
    # Calcular métricas
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    mixed_images = []
    
    if COUNT_UNIQUE_FRAMES:
        # Opção A: Contar frames únicos
        # Agrupar crops por frame
        frames_falso = set((sample, frame) for sample, frame, person in crops_falso)
        frames_real = set((sample, frame) for sample, frame, person in crops_real)
        weapon_frames = set((sample, frame) for sample, frame, person in weapon_crops)
        
        # True Negatives: frames em crops/falso*/* que não têm arma
        for frame in frames_falso:
            if frame not in weapon_frames:
                true_negatives += 1
        
        # False Negatives: frames em crops/real*/* que não têm arma
        for frame in frames_real:
            if frame not in weapon_frames:
                false_negatives += 1
        
        # False Positives: frames com detecções em weapon_detections/falso*/*
        falso_weapon_frames = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("falso"):
                for (frame_num, person_id) in detections.keys():
                    falso_weapon_frames.add((sample, frame_num))
        
        false_positives += len(falso_weapon_frames)
        
        # False Positives: frames com detecções anotadas manualmente
        manual_fp_frames = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("real"):
                for (frame_num, person_id), crops in detections.items():
                    for detection in crops:
                        if detection['full_path'] in MANUAL_FALSE_POSITIVES:
                            manual_fp_frames.add((sample, frame_num))
                            break
        
        # True Positives: frames com detecções em weapon_detections/real*/* (exceto só manuais)
        real_weapon_frames = set()
        mixed_frames = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("real"):
                for (frame_num, person_id), crops in detections.items():
                    has_non_manual = False
                    for detection in crops:
                        if detection['full_path'] not in MANUAL_FALSE_POSITIVES:
                            has_non_manual = True
                            break
                    
                    frame_key = (sample, frame_num)
                    if has_non_manual:
                        real_weapon_frames.add(frame_key)
                    
                    # Verificar se é mixed
                    crop_key = (sample, frame_num, person_id)
                    if crop_key in weapon_crops_with_manual_fp and weapon_crops_with_manual_fp[crop_key]['other_detections']:
                        mixed_frames.add(frame_key)
        
        # Remover mixed de FP manual
        manual_fp_frames = manual_fp_frames - mixed_frames
        
        false_positives += len(manual_fp_frames)
        true_positives = len(real_weapon_frames)
        
        # MIXED_IMAGES
        for crop_key, info in weapon_crops_with_manual_fp.items():
            if info['other_detections']:
                frame_key = (crop_key[0], crop_key[1])
                if frame_key in mixed_frames:
                    mixed_images.append({
                        'sample': crop_key[0],
                        'frame': crop_key[1],
                        'person': crop_key[2],
                        'manual_fps': info['manual_fps'],
                        'other_detections': info['other_detections']
                    })
    else:
        # Opção B: Contar crops individuais
        # True Negatives: crops em crops/falso*/* que não têm arma
        for crop in crops_falso:
            if crop not in weapon_crops:
                true_negatives += 1
        
        # False Negatives: crops em crops/real*/* que não têm arma
        for crop in crops_real:
            if crop not in weapon_crops:
                false_negatives += 1
        
        # False Positives: crops com detecções em weapon_detections/falso*/*
        falso_weapon_crops = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("falso"):
                for (frame_num, person_id) in detections.keys():
                    falso_weapon_crops.add((sample, frame_num, person_id))
        
        false_positives += len(falso_weapon_crops)
        
        # False Positives: crops com detecções anotadas manualmente (apenas manuais)
        manual_fp_crops = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("real"):
                for (frame_num, person_id), crops in detections.items():
                    has_manual = False
                    has_non_manual = False
                    
                    for detection in crops:
                        if detection['full_path'] in MANUAL_FALSE_POSITIVES:
                            has_manual = True
                        else:
                            has_non_manual = True
                    
                    crop_key = (sample, frame_num, person_id)
                    # Se tem manual e não tem outros, é FP
                    if has_manual and not has_non_manual:
                        manual_fp_crops.add(crop_key)
        
        false_positives += len(manual_fp_crops)
        
        # True Positives: crops com detecções válidas em weapon_detections/real*/*
        real_weapon_crops = set()
        for sample, detections in weapon_detections.items():
            if sample.startswith("real"):
                for (frame_num, person_id), crops in detections.items():
                    has_non_manual = False
                    for detection in crops:
                        if detection['full_path'] not in MANUAL_FALSE_POSITIVES:
                            has_non_manual = True
                            break
                    
                    if has_non_manual:
                        crop_key = (sample, frame_num, person_id)
                        real_weapon_crops.add(crop_key)
        
        true_positives = len(real_weapon_crops)
        
        # MIXED_IMAGES: crops que têm tanto detecções manuais quanto válidas
        for crop_key, info in weapon_crops_with_manual_fp.items():
            if info['other_detections']:
                mixed_images.append({
                    'sample': crop_key[0],
                    'frame': crop_key[1],
                    'person': crop_key[2],
                    'manual_fps': info['manual_fps'],
                    'other_detections': info['other_detections']
                })
    
    return {
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_positives': true_positives,
        'mixed_images': mixed_images
    }


def analyze_by_sample(weapon_dir, confidence_threshold, fraction_threshold):
    """Analisa métricas por sample."""
    # Obter todas as detecções (não apenas as acima do threshold de confiança)
    # porque precisamos verificar quais frames têm APENAS crops manuais
    all_detections = get_weapon_detections_by_confidence(weapon_dir, 0.0)
    weapon_detections = get_weapon_detections_by_confidence(weapon_dir, confidence_threshold)
    
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    
    for sample_name, person_count in PERSON_DETECTIONS_PER_SAMPLE.items():
        # Contar frames com detecção de arma (ignorando manuais)
        frames_with_weapon = set()
        
        # Ajustar person_count: remover frames que têm APENAS crops manuais
        adjusted_person_count = person_count
        
        if sample_name in all_detections:
            for (frame_num, person_id), crops in all_detections[sample_name].items():
                # Verificar se o crop tem APENAS detecções manuais
                all_manual = True
                for detection in crops:
                    if detection['full_path'] not in MANUAL_FALSE_POSITIVES:
                        all_manual = False
                        break
                
                # Se todas as detecções deste crop são manuais, reduzir person_count
                if all_manual:
                    adjusted_person_count -= 1
        
        if sample_name in weapon_detections:
            for (frame_num, person_id), crops in weapon_detections[sample_name].items():
                # Verificar se tem alguma detecção que não é manual
                has_non_manual = False
                for detection in crops:
                    if detection['full_path'] not in MANUAL_FALSE_POSITIVES:
                        has_non_manual = True
                        break
                
                if has_non_manual:
                    frames_with_weapon.add(frame_num)
        
        # Calcular fração com person_count ajustado
        if adjusted_person_count > 0:
            fraction = len(frames_with_weapon) / adjusted_person_count
        else:
            fraction = 0
        
        # Classificar o sample
        is_positive_detection = fraction >= fraction_threshold
        is_real_weapon = sample_name.startswith("real")
        
        if is_real_weapon:
            if is_positive_detection:
                true_positives += 1
            else:
                false_negatives += 1
        else:  # falso
            if is_positive_detection:
                false_positives += 1
            else:
                true_negatives += 1
    
    return {
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_positives': true_positives
    }


def calculate_metrics(tn, fn, fp, tp):
    """Calcula métricas de avaliação."""
    # Total de samples/frames
    total = tn + fn + fp + tp
    
    # Accuracy: (TP + TN) / Total
    if total > 0:
        accuracy = (tp + tn) / total
    else:
        accuracy = 0.0
    
    # Precision: TP / (TP + FP)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
    
    # Recall: TP / (TP + FN)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0
    
    # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def validate_metrics(validation_mode='full'):
    """
    Valida as métricas calculadas.
    
    Args:
        validation_mode: 'full' (mostra tudo), 'errors' (só erros), 'silent' (suprime saída)
    
    Returns:
        bool: True se todas as validações passaram, False caso contrário
    """
    mode_description = "FRAMES ÚNICOS (Opção A)" if COUNT_UNIQUE_FRAMES else "CROPS INDIVIDUAIS (Opção B)"
    
    # Totais esperados
    total_samples = 24  # 12 falso + 12 real
    total_crops = 238  # Total de detecções de pessoa
    total_frames_unicos = 236  # Frames únicos (2 frames têm 2 pessoas)
    expected_frame_total = total_frames_unicos if COUNT_UNIQUE_FRAMES else total_crops
    
    all_passed = True
    errors = []
    
    if validation_mode == 'full':
        print("\n" + "=" * 80)
        print("VALIDAÇÃO DE MÉTRICAS")
        print("=" * 80)
        print(f"Modo de contagem: {mode_description}")
        print("=" * 80)
        print()
        print(f"Totais esperados:")
        print(f"  Samples: {total_samples}")
        print(f"  Crops (detecções de pessoa): {total_crops}")
        print(f"  Frames únicos: {total_frames_unicos}")
        print(f"  Total esperado neste modo: {expected_frame_total}")
        print()
    
    # Validar análise por frame/crop
    if validation_mode == 'full':
        print("=" * 80)
        print("VALIDAÇÃO: ANÁLISE POR FRAME/CROP")
        print("=" * 80)
        print()
    
    for conf_threshold in CONFIDENCE_THRESHOLDS:
        results = analyze_by_frame(CROPS_DIR, WEAPON_DETECTIONS_DIR, conf_threshold)
        
        tn = results['true_negatives']
        fn = results['false_negatives']
        fp = results['false_positives']
        tp = results['true_positives']
        total = tn + fn + fp + tp
        
        metrics = calculate_metrics(tn, fn, fp, tp)
        
        passed = (total == expected_frame_total)
        if not passed:
            all_passed = False
            error_msg = f"Frame/Crop (Conf >= {conf_threshold}): Total={total}, Esperado={expected_frame_total}"
            errors.append(error_msg)
            if validation_mode == 'errors':
                print(f"✗ ERRO: {error_msg}")
        
        if validation_mode == 'full':
            status = "✓" if passed else "✗ ERRO"
            print(f"Confiança >= {conf_threshold}:")
            print(f"  TN={tn:3d} | FN={fn:3d} | FP={fp:3d} | TP={tp:3d} | Total={total}/{expected_frame_total} {status}")
            print(f"  Acc={metrics['accuracy']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | F1={metrics['f1_score']:.4f}")
            print()
    
    # Validar análise por sample
    if validation_mode == 'full':
        print("=" * 80)
        print("VALIDAÇÃO: ANÁLISE POR SAMPLE")
        print("=" * 80)
        print()
    
    for frac_threshold in FRACTION_THRESHOLDS:
        if validation_mode == 'full':
            print(f"--- Fração >= {frac_threshold} ---")
        
        for conf_threshold in CONFIDENCE_THRESHOLDS:
            results = analyze_by_sample(WEAPON_DETECTIONS_DIR, conf_threshold, frac_threshold)
            
            tn = results['true_negatives']
            fn = results['false_negatives']
            fp = results['false_positives']
            tp = results['true_positives']
            total = tn + fn + fp + tp
            
            passed = (total == total_samples)
            if not passed:
                all_passed = False
                error_msg = f"Sample (Frac >= {frac_threshold}, Conf >= {conf_threshold}): Total={total}, Esperado={total_samples}"
                errors.append(error_msg)
                if validation_mode == 'errors':
                    print(f"✗ ERRO: {error_msg}")
            
            if validation_mode == 'full':
                status = "✓" if passed else "✗ ERRO"
                print(f"  Conf >= {conf_threshold}: TN={tn:2d} | FN={fn:2d} | FP={fp:2d} | TP={tp:2d} | Total={total}/{total_samples} {status}")
        
        if validation_mode == 'full':
            print()
    
    if validation_mode == 'full':
        print("=" * 80)
        if all_passed:
            print("✓ TODAS AS VALIDAÇÕES PASSARAM")
        else:
            print("✗ ALGUMAS VALIDAÇÕES FALHARAM")
            print(f"Total de erros: {len(errors)}")
        print("=" * 80)
    elif validation_mode == 'errors' and all_passed:
        print("✓ Todas as validações passaram (nenhum erro encontrado)")
    
    return all_passed


def main():
    """
    Executa a análise de métricas com opções configuráveis.
    
    Uso:
        python analyze_metrics.py [opção] [validação]
    
    Parâmetros:
        opção: 'A' ou 'a' para frames únicos (padrão), 'B' ou 'b' para crops individuais
        validação: 'full' (validação completa), 'errors' (só erros), 'silent' (sem validação), 'none' (sem validação)
    
    Exemplos:
        python analyze_metrics.py A full       # Opção A com validação completa
        python analyze_metrics.py B errors     # Opção B mostrando só erros
        python analyze_metrics.py A silent     # Opção A sem validação
        python analyze_metrics.py              # Opção A com validação completa (padrão)
    """
    global COUNT_UNIQUE_FRAMES
    
    # Processar argumentos da linha de comando
    option = 'A'  # Padrão: Opção A (frames únicos)
    validation_mode = 'full'  # Padrão: validação completa
    
    if len(sys.argv) > 1:
        option = sys.argv[1].upper()
        if option not in ['A', 'B']:
            print(f"Erro: Opção inválida '{sys.argv[1]}'. Use 'A' ou 'B'.")
            print()
            print("Uso: python analyze_metrics.py [opção] [validação]")
            print("  opção: 'A' (frames únicos) ou 'B' (crops individuais)")
            print("  validação: 'full', 'errors', 'silent', 'none'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        validation_mode = sys.argv[2].lower()
        if validation_mode not in ['full', 'errors', 'silent', 'none']:
            print(f"Erro: Modo de validação inválido '{sys.argv[2]}'. Use 'full', 'errors', 'silent' ou 'none'.")
            print()
            print("Uso: python analyze_metrics.py [opção] [validação]")
            print("  opção: 'A' (frames únicos) ou 'B' (crops individuais)")
            print("  validação: 'full', 'errors', 'silent', 'none'")
            sys.exit(1)
    
    # Configurar COUNT_UNIQUE_FRAMES baseado na opção
    COUNT_UNIQUE_FRAMES = (option == 'A')
    
    mode_description = "FRAMES ÚNICOS (Opção A)" if COUNT_UNIQUE_FRAMES else "CROPS INDIVIDUAIS (Opção B)"
    
    # Preparar listas para armazenar resultados
    frame_results = []
    sample_results = []
    
    mode_description = "FRAMES ÚNICOS (Opção A)" if COUNT_UNIQUE_FRAMES else "CROPS INDIVIDUAIS (Opção B)"
    print("=" * 80)
    print(f"ANÁLISE POR FRAME - Modo: {mode_description}")
    print("=" * 80)
    
    for conf_threshold in CONFIDENCE_THRESHOLDS:
        print(f"\n--- Confiança >= {conf_threshold} ---")
        results = analyze_by_frame(CROPS_DIR, WEAPON_DETECTIONS_DIR, conf_threshold)
        
        print(f"True Negatives:  {results['true_negatives']}")
        print(f"False Negatives: {results['false_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"True Positives:  {results['true_positives']}")
        
        # Calcular e exibir métricas
        metrics = calculate_metrics(
            results['true_negatives'],
            results['false_negatives'],
            results['false_positives'],
            results['true_positives']
        )
        
        print(f"\nMétricas:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Armazenar resultados para Excel
        frame_results.append({
            'Modo': mode_description,
            'Confiança Mínima': conf_threshold,
            'True Negatives': results['true_negatives'],
            'False Negatives': results['false_negatives'],
            'False Positives': results['false_positives'],
            'True Positives': results['true_positives'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    print("\n" + "=" * 80)
    print("ANÁLISE POR SAMPLE")
    print("=" * 80)
    
    for frac_threshold in FRACTION_THRESHOLDS:
        print(f"\n### Fração mínima >= {frac_threshold} ###")
        
        for conf_threshold in CONFIDENCE_THRESHOLDS:
            print(f"\n  --- Confiança >= {conf_threshold} ---")
            results = analyze_by_sample(WEAPON_DETECTIONS_DIR, conf_threshold, frac_threshold)
            
            print(f"  True Negatives:  {results['true_negatives']}")
            print(f"  False Negatives: {results['false_negatives']}")
            print(f"  False Positives: {results['false_positives']}")
            print(f"  True Positives:  {results['true_positives']}")
            
            # Calcular e exibir métricas
            metrics = calculate_metrics(
                results['true_negatives'],
                results['false_negatives'],
                results['false_positives'],
                results['true_positives']
            )
            
            print(f"\n  Métricas:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1_score']:.4f}")
            
            # Armazenar resultados para Excel
            sample_results.append({
                'Modo': mode_description,
                'Fração Mínima': frac_threshold,
                'Confiança Mínima': conf_threshold,
                'True Negatives': results['true_negatives'],
                'False Negatives': results['false_negatives'],
                'False Positives': results['false_positives'],
                'True Positives': results['true_positives'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
    
    # Executar validação se não for 'none' ou 'silent'
    if validation_mode not in ['none', 'silent']:
        validate_metrics(validation_mode)
    elif validation_mode == 'silent':
        # Executa validação mas sem output
        validate_metrics('silent')
    
    # Salvar resultados em Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metricas_analise_{option}_{timestamp}.xlsx"
    
    print("\n" + "=" * 80)
    print(f"Salvando resultados em: {filename}")
    print("=" * 80)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Salvar análise por frame
        df_frame = pd.DataFrame(frame_results)
        df_frame.to_excel(writer, sheet_name='Análise por Frame', index=False)
        
        # Salvar análise por sample
        df_sample = pd.DataFrame(sample_results)
        df_sample.to_excel(writer, sheet_name='Análise por Sample', index=False)
    
    print(f"✓ Arquivo Excel salvo com sucesso!")
    print(f"  - Análise por Frame: {len(frame_results)} configurações")
    print(f"  - Análise por Sample: {len(sample_results)} configurações")
    print("=" * 80)


if __name__ == "__main__":
    main()
