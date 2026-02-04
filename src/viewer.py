import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
from pathlib import Path

# cores
color_text_body = (255, 255, 255)
color_text_title = (226, 135, 67)  # laranja
color_text_weapon = (255, 100, 100)  # vermelho claro

color_rect_person = (0, 180, 0)  # verde para pessoa
color_rect_weapon = (0, 0, 255)  # vermelho para arma
color_rect_bg = (0, 0, 0)  # fundo preto

color_pin = (67, 135, 226)  # azul
color_pin_weapon = (255, 100, 100)  # vermelho claro

min_rect_width = 400
show_bb_track = True

# fonte
font_size = 12
font = None
font_small = None
font_path_global = None  # armazena caminho da fonte para escala dinamica

def _load_fonts():
    """carrega fontes ttf para texto de alta qualidade"""
    global font, font_small, font_path_global
    
    if font is not None:
        return
    
    # tenta encontrar arial.ttf
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        'C:\\Windows\\Fonts\\arial.ttf',
    ]
    
    font_path_global = None
    for path in font_paths:
        if os.path.isfile(path):
            font_path_global = path
            break
    
    if font_path_global:
        try:
            font = ImageFont.truetype(font_path_global, size=font_size)
            font_small = ImageFont.truetype(font_path_global, size=16)
            return
        except Exception:
            pass
    
    # fallback para fonte default
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()


def draw_texts(source_image, values):
    """desenha textos usando PIL para melhor qualidade com escala opcional"""
    _load_fonts()
    
    image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    for value in values:
        # suporta formato antigo (5 elementos) e novo (6 elementos com scale)
        if len(value) == 6:
            text, x, y, color, use_small_font, scale_factor = value
            # cria fonte escalada dinamicamente
            scaled_size = int(font_size * scale_factor)
            if font_path_global:
                try:
                    scaled_font = ImageFont.truetype(font_path_global, size=scaled_size)
                except:
                    scaled_font = font  # fallback
            else:
                scaled_font = font  # fallback
            current_font = scaled_font
        else:
            text, x, y, color, use_small_font = value
            current_font = font_small if use_small_font else font
        
        draw = ImageDraw.Draw(pil_image)
        draw.text((x, y), text, fill=color, font=current_font)
    
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def tracks_from_detections(detections_info, weapon_results=None, track_id_start=1):
    """Convert pipeline detections + optional weapon results into viewer track dicts.

    This lets batch pipelines reuse the same overlay code used by tracking mode.

    Args:
        detections_info: list[dict] with at least 'bbox' in xyxy pixel coords.
        weapon_results: optional list aligned with detections (per-person crop results).
        track_id_start: starting id for synthetic track ids.

    Returns:
        list[dict] compatible with `_extract_track_data` (dict branch).
    """

    tracks = []
    detections_info = detections_info or []
    weapon_results = weapon_results or []

    for i, det in enumerate(detections_info):
        if not isinstance(det, dict):
            continue

        bbox = det.get('bbox')
        if bbox is None:
            continue

        # Choose a single distance value for display.
        distance = det.get('distance_m')
        if distance is None:
            distance = det.get('distance_pitch_m')
        if distance is None:
            distance = det.get('distance_pinhole_m')

        # Geo fields come from different keys depending on pipeline.
        lat = det.get('lat')
        lon = det.get('lon')
        if lat is None or lon is None:
            geo = det.get('person_geoposition') or det.get('fused_geoposition')
            if isinstance(geo, dict):
                lat = geo.get('latitude', lat)
                lon = geo.get('longitude', lon)

        has_weapon = bool(det.get('has_weapon', False))
        weapon_conf = float(det.get('weapon_confidence', 0.0) or 0.0)

        # If weapon results are supplied, compute weapon status/confidence from them.
        if i < len(weapon_results) and isinstance(weapon_results[i], dict):
            wr = weapon_results[i]
            has_weapon = bool(wr.get('has_weapons', has_weapon))
            if has_weapon:
                w_dets = wr.get('weapon_detections') or []
                if w_dets:
                    try:
                        weapon_conf = float(max([w.get('weapon_confidence', w.get('confidence', 0.0)) for w in w_dets]))
                    except Exception:
                        # keep best-effort weapon_conf
                        pass

        track_id = det.get('track_id', track_id_start + i)

        tracks.append({
            'track_id': track_id,
            'bbox': bbox,
            'has_weapon': has_weapon,
            'weapon_confidence': weapon_conf,
            'weapon_avg_confidence': weapon_conf,
            'weapon_peak_confidence': weapon_conf,
            'temporal_voting_active': bool(det.get('temporal_voting_active', False)),
            'weapon_lost': bool(det.get('weapon_lost', False)),
            'distance': distance,
            'bearing': det.get('bearing_deg', det.get('bearing')),
            'lat': lat,
            'lon': lon,
            'x_utm': det.get('x_utm'),
            'y_utm': det.get('y_utm'),
            'weapon_bboxes': det.get('weapon_bboxes', []),
            'lost': bool(det.get('lost', False)),
        })

    return tracks


def draw_info_panel(
    image,
    lines,
    x,
    y,
    scale_factor=1.0,
    align='left',
    bg_color=(0, 0, 0),
    bg_opacity=0.8,
):
    """Draw a multi-line info panel with translucent background.

    Args:
        image: BGR image.
        lines: list of (text, color_bgr) tuples.
        x, y: anchor position in pixels. For align='left' it's top-left.
        scale_factor: multiplies font size.
        align: 'left' or 'center'.
        bg_color: BGR background fill.
        bg_opacity: background opacity [0..1].

    Returns:
        (image, (x1, y1, x2, y2)) panel rectangle.
    """
    _load_fonts()

    if image is None:
        return image, (0, 0, 0, 0)

    if not lines:
        return image, (int(x), int(y), int(x), int(y))

    h_img, w_img = image.shape[:2]
    scale_factor = float(scale_factor) if scale_factor is not None else 1.0
    scale_factor = max(0.5, min(2.5, scale_factor))

    # Approximate text metrics.
    scaled_font_size = int(font_size * scale_factor)
    line_spacing = int((font_size + 6) * scale_factor)
    pad_x = int(10 * scale_factor)
    pad_y = int(8 * scale_factor)

    max_text_length = max(len(str(t[0])) for t in lines)
    panel_w = int(max_text_length * scaled_font_size * 0.6) + 2 * pad_x
    panel_h = len(lines) * line_spacing + 2 * pad_y

    if align == 'center':
        x1 = int(x - panel_w // 2)
    else:
        x1 = int(x)
    y1 = int(y)
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    # Clamp to image bounds.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w_img:
        x1 -= (x2 - w_img)
        x2 = w_img
        x1 = max(0, x1)
    if y2 > h_img:
        y1 -= (y2 - h_img)
        y2 = h_img
        y1 = max(0, y1)

    # Draw translucent background.
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, float(bg_opacity), image, 1.0 - float(bg_opacity), 0, image)

    # Prepare PIL text draws.
    text_values = []
    text_x = x1 + pad_x
    text_y = y1 + pad_y
    for text, color in lines:
        use_small = scale_factor < 0.75
        text_values.append([str(text), int(text_x), int(text_y), tuple(color), use_small, scale_factor])
        text_y += line_spacing

    image = draw_texts(image, text_values)
    return image, (x1, y1, x2, y2)


def _extract_track_data(track):
    """extrai dados do track (suporta PersonTrack e dict)"""
    
    # objeto PersonTrack
    if not isinstance(track, dict):
        bbox_xywh = track.bbox
        if bbox_xywh is None or track.lost:
            return None
        
        x, y, w, h = bbox_xywh
        bbox = [x, y, x + w, y + h]
        
        has_weapon = track.weapon_classifier.has_weapon()
        
        # obtem confianca correta baseado no modo de votacao
        weapon_conf = 0.0
        temporal_voting_active = False
        
        classifier = track.weapon_classifier
        
        # SEMPRE pega a confianca atual (agora sempre atualizado)
        weapon_conf = classifier.current_confidence
        
        # Pega media e pico do temporal voting
        weapon_avg_conf = classifier.categories['armed']['confidence']  # media
        weapon_peak_conf = classifier.categories['armed'].get('peak', weapon_avg_conf)  # pico
        
        # Temporal voting esta ativo se o sistema estiver configurado para usar
        temporal_voting_active = classifier.use_temporal_voting
        
        # usa weapon_tracks com kalman filter se disponivel, senao usa weapon_bboxes raw
        weapon_bboxes = []
        max_weapon_conf = 0.0  # track da maior confianca dos weapon tracks
        
        if track.weapon_tracks:
            # usa bboxes suavizados pelo kalman filter
            for wt in track.weapon_tracks:
                # inclui weapon bboxes apenas se track nao estiver perdido
                if not wt.lost:
                    weapon_bboxes.append({
                        'bbox': wt.get_bbox('xyxy'),
                        'confidence': wt.confidence,
                        'class': wt.weapon_class,
                        'smoothed': True
                    })
                
                # mas considera a confianca MESMO se track estiver perdido
                # (para manter exibicao do texto de confianca)
                if wt.confidence > max_weapon_conf:
                    max_weapon_conf = wt.confidence
            
            # se temos weapon tracks (mesmo perdidos) com confianca, mostra arma
            if max_weapon_conf > 0:
                has_weapon = True  # força exibição
                weapon_conf = max_weapon_conf
                
        elif track.weapon_bboxes:
            # fallback para bboxes raw
            for wb in track.weapon_bboxes:
                if isinstance(wb, dict):
                    weapon_bboxes.append({
                        'bbox': wb.get('bbox', wb.get('bbox_crop', [])),
                        'confidence': wb.get('confidence', 0.0),
                        'class': wb.get('class', 'weapon'),
                        'smoothed': False
                    })
                else:
                    weapon_bboxes.append({
                        'bbox': wb,
                        'confidence': 0.0,
                        'class': 'weapon',
                        'smoothed': False
                    })
        
        weapon_lost = False
        
        if has_weapon and track.weapon_tracks:
            active_tracks = sum(1 for wt in track.weapon_tracks if wt.frames_since_update == 0)
            total_tracks = len(track.weapon_tracks)
            frames_info = [wt.frames_since_update for wt in track.weapon_tracks]
            
            # Se tem weapon tracks mas nenhum esta recebendo deteccao ativa neste frame
            weapon_lost = (total_tracks > 0 and active_tracks == 0)
        
        return {
            'track_id': track.id,
            'bbox': bbox,
            'has_weapon': has_weapon,
            'weapon_confidence': weapon_conf,
            'weapon_avg_confidence': weapon_avg_conf if weapon_avg_conf > 0 else weapon_conf,
            'weapon_peak_confidence': weapon_peak_conf if weapon_peak_conf > 0 else weapon_conf,
            'temporal_voting_active': temporal_voting_active,
            'weapon_lost': weapon_lost,
            'distance': track.distance,
            'bearing': track.bearing,
            'lat': track.lat,
            'lon': track.lon,
            'x_utm': track.x_utm,
            'y_utm': track.y_utm,
            'weapon_bboxes': weapon_bboxes,
            'lost': track.lost
        }
    
    # formato dict
    else:
        bbox = track.get('bbox')
        if bbox is None:
            return None
        
        weapon_conf = track.get('weapon_confidence', 0.0)
        return {
            'track_id': track.get('track_id', 'Unknown'),
            'bbox': bbox,
            'has_weapon': track.get('has_weapon', False),
            'weapon_confidence': weapon_conf,
            'weapon_avg_confidence': track.get('weapon_avg_confidence', weapon_conf),
            'weapon_peak_confidence': track.get('weapon_peak_confidence', weapon_conf),
            'temporal_voting_active': track.get('temporal_voting_active', False),
            'weapon_lost': track.get('weapon_lost', False),
            'distance': track.get('distance'),
            'bearing': track.get('bearing'),
            'lat': track.get('lat'),
            'lon': track.get('lon'),
            'x_utm': track.get('x_utm'),
            'y_utm': track.get('y_utm'),
            'weapon_bboxes': track.get('weapon_bboxes', []),
            'lost': track.get('lost', False)
        }
    
    return None


def _draw_person_overlay(image, track_data, show_confidence=False):
    bbox = track_data['bbox']
    track_id = track_data['track_id']
    distance = track_data['distance']
    lat = track_data['lat']
    lon = track_data['lon']
    has_weapon = track_data['has_weapon']
    weapon_conf = track_data['weapon_confidence']
    weapon_avg_conf = track_data.get('weapon_avg_confidence', weapon_conf)
    weapon_peak_conf = track_data.get('weapon_peak_confidence', weapon_conf)
    temporal_voting_active = track_data.get('temporal_voting_active', False)
    weapon_lost = track_data.get('weapon_lost', False)
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
    except (ValueError, TypeError, IndexError):
        return image, []
    
    text_values = []
    
    # determina cor da bbox baseado na confianca da ARMA (weapon)
    if has_weapon and weapon_conf > 0.2:
        # deteccao atual forte - VERMELHO
        bbox_color = (0, 0, 255)
    elif has_weapon and temporal_voting_active:
        # apenas temporal voting (memoria) - LARANJA
        bbox_color = (67, 135, 226)
    else:
        bbox_color = color_rect_person
    
    image_height = image.shape[0]
    resolution_scale = image_height / 720.0
    
    # calcula escala baseada no tamanho da bbox (maior bbox = texto maior)
    # tamanho base: 100px de altura = scale 1.0
    bbox_height = h
    bbox_scale = bbox_height / 100.0
    
    # combina ambas as escalas (resolucao tem peso maior)
    # minimo: 0.5x, maximo: 2.5x
    scale_factor = max(0.5, min(2.5, resolution_scale * 0.7 + bbox_scale * 0.3))
    scaled_font_size = int(font_size * scale_factor)
    scaled_line_spacing = int((font_size + 3) * scale_factor)
    
    # desenha apenas a bbox da pessoa (sem box de info, sem linha, sem circulo)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
    
    # prepara todas as linhas de texto com suas cores
    text_data = []
    
    # titulo (laranja)
    text_data.append((f"ID: {track_id}", color_text_title))
    
    # distancia (branco)
    if distance is not None:
        text_data.append((f"Distância: {distance:.1f}m", color_text_body))
    
    # coordenadas geograficas (branco)
    if lat is not None and lon is not None:
        text_data.append((f"Lat:{lat:.6f} Lon:{lon:.6f}", color_text_body))
    
    # Flag de arma com cor baseada no tipo de deteccao
    if has_weapon:
        # vermelho: deteccao atual forte (weapon_conf > 40%)
        # laranja: apenas temporal voting (deteccao temporaria/memoria)
        if weapon_conf > 0.2:
            armado_color = color_text_weapon  # (255, 100, 100) vermelho claro
        else:
            armado_color = color_text_title   # (226, 135, 67) laranja
        
        text_data.append(("ARMADO", armado_color))
        
        # Mostra confiança apenas se show_confidence=True
        if show_confidence:
            current_pct = int(weapon_conf * 100)
            avg_pct = int(weapon_avg_conf * 100)
            peak_pct = int(weapon_peak_conf * 100)
            
            if temporal_voting_active:
                # Sistema usa temporal voting - mostra current/avg/peak
                text_data.append((f"Confiança: {current_pct}%/{avg_pct}%/{peak_pct}%", color_text_weapon))
            elif weapon_lost:
                text_data.append((f"Confiança: {current_pct}% (?)", color_text_weapon))
            else:
                # Sem temporal voting - mostra apenas confianca atual
                text_data.append((f"Confiança: {current_pct}%", color_text_weapon))
    
    # calcula dimensoes do background com escala
    num_lines = len(text_data)
    if num_lines > 0:
        # estima largura maxima do texto (aproximacao com escala)
        max_text_length = max(len(text) for text, _ in text_data)
        bg_width = int(max_text_length * scaled_font_size * 0.6) + int(20 * scale_factor)  # padding escalado
        bg_height = num_lines * scaled_line_spacing + int(10 * scale_factor)  # padding vertical escalado
        
        # calcula posicao centralizada em relacao ao topo da bbox
        bbox_center_x = x1 + w // 2
        bg_x1 = bbox_center_x - bg_width // 2
        bg_x2 = bg_x1 + bg_width
        
        # posiciona acima da bbox
        bg_y2 = y1 - 5
        bg_y1 = bg_y2 - bg_height
        
        # garante que nao saia da tela
        if bg_y1 < 0:
            bg_y1 = 0
            bg_y2 = bg_height
        
        # desenha background semi-transparente
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)  # 80% opacity

        # calcula posicao inicial do texto (no topo do background com padding escalado)
        text_x = bg_x1 + int(10 * scale_factor)
        line_y = bg_y1 + int(5 * scale_factor)  # pequeno padding do topo escalado
        
        # adiciona todos os textos com escala
        for text, color in text_data:
            # usa small font se escala for pequena
            use_small = scale_factor < 0.75
            text_values.append([text, text_x, line_y, color, use_small, scale_factor])
            line_y += scaled_line_spacing
    else:
        # fallback se nao houver texto
        text_x = x1
        line_y = max(y1 - 10, 25)
    
    return image, text_values

def draw_bbox(frame, tracks, show_confidence=False):
    """
    Desenha bounding boxes e informações nos tracks.
    
    Args:
        frame: imagem do frame
        tracks: lista de tracks
        show_confidence: se True, mostra confiança de arma (default: False)
    """
    if frame is None or not tracks:
        return frame
    
    image = frame.copy()
    text_values = []
    
    # processa cada track
    for track in tracks:
        track_data = _extract_track_data(track)
        
        if track_data is None or track_data['lost']:
            continue
        
        # desenha overlay da pessoa (apenas bbox e texto, sem weapon bbox)
        image, person_texts = _draw_person_overlay(image, track_data, show_confidence)
        text_values.extend(person_texts)
    
    # desenha todos os textos com PIL (alta qualidade)
    if text_values:
        image = draw_texts(image, text_values)
    
    return image