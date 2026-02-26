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
        if distance is None:
            distance = det.get('distance_fused_m')

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
            has_weapon = bool(wr.get('has_weapon', has_weapon))
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


def _draw_person_overlay(image, track_data, extra_lines=None):
    bbox = track_data['bbox']
    track_id = track_data['track_id']
    distance = track_data['distance']
    lat = track_data['lat']
    lon = track_data['lon']
    has_weapon = track_data['has_weapon']
    weapon_conf = track_data['weapon_confidence']
    
    try:
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
    except (ValueError, TypeError, IndexError):
        return image, []
    
    text_values = []

    if has_weapon and weapon_conf > 0.2:
        bbox_color = (0, 0, 255)
    else:
        bbox_color = color_rect_person
    
    image_height = image.shape[0]
    resolution_scale = image_height / 720.0
    
    bbox_height = h
    bbox_scale = bbox_height / 100.0
    
    scale_factor = max(0.5, min(2.5, resolution_scale * 0.7 + bbox_scale * 0.3))
    scaled_font_size = int(font_size * scale_factor)
    scaled_line_spacing = int((font_size + 3) * scale_factor)
    
    image = cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
    
    text_data = []
    
    text_data.append((f"ID: {track_id}", color_text_title))
    if distance is not None:
        text_data.append((f"Distance: {distance:.1f}m", color_text_body))
    if lat is not None and lon is not None:
        text_data.append((f"Lat:{lat:.6f} Lon:{lon:.6f}", color_text_body))
    if has_weapon:
        text_data.append((f"Confidence: {weapon_conf:.3f}", color_text_weapon))
    if extra_lines:
        for line_text, line_color in extra_lines:
            text_data.append((line_text, line_color))

    num_lines = len(text_data)
    if num_lines > 0:
        max_text_length = max(len(text) for text, _ in text_data)
        bg_width = int(max_text_length * scaled_font_size * 0.6) + int(20 * scale_factor)  # padding escalado
        bg_height = num_lines * scaled_line_spacing + int(10 * scale_factor)  # padding vertical escalado
        bbox_center_x = x1 + w // 2
        bg_x1 = bbox_center_x - bg_width // 2
        bg_x2 = bg_x1 + bg_width
        bg_y2 = y1 - 5
        bg_y1 = bg_y2 - bg_height
        if bg_y1 < 0:
            bg_y1 = 0
            bg_y2 = bg_height
        
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)  # 80% opacity

        text_x = bg_x1 + int(10 * scale_factor)
        line_y = bg_y1 + int(5 * scale_factor)  # pequeno padding do topo escalado
        
        for text, color in text_data:
            use_small = scale_factor < 0.75
            text_values.append([text, text_x, line_y, color, use_small, scale_factor])
            line_y += scaled_line_spacing
    else:
        # fallback se nao houver texto
        text_x = x1
        line_y = max(y1 - 10, 25)
    
    return image, text_values

def draw_bbox(frame, tracks, show_confidence=False, tracks_extra_lines=None):
    if frame is None or not tracks:
        return frame
    
    image = frame.copy()
    text_values = []
    
    tracks_extra_lines = tracks_extra_lines or {}
    
    # processa cada track
    for idx, track in enumerate(tracks):
        track_data = _extract_track_data(track)
        
        if track_data is None or track_data['lost']:
            continue
        
        extra = tracks_extra_lines.get(idx)
        
        # desenha overlay da pessoa (apenas bbox e texto, sem weapon bbox)
        image, person_texts = _draw_person_overlay(image, track_data, extra_lines=extra)
        text_values.extend(person_texts)
    
    # desenha todos os textos com PIL (alta qualidade)
    if text_values:
        image = draw_texts(image, text_values)
    
    return image

def draw_boxes_fusion(image, detections, weapon_results, drone_label, fused_detections=None, drone_id=None, show_confidence=False):
    tracks = tracks_from_detections(detections, weapon_results, track_id_start=1)
    # ...existing code...
    # Fused info commented out: only single detection info is shown
    tracks_extra_lines = {}
    # if fused_detections and drone_id is not None:
    #     bbox_key = f'bbox_drone{drone_id}'
    #     for track_idx, det in enumerate(detections):
    #         det_bbox = det.get('bbox')
    #         if det_bbox is None:
    #             continue
    #         # Find matching fused detections (both methods)
    #         matched_fused = []
    #         for fd in fused_detections:
    #             fused_bbox = fd.get(bbox_key) or fd.get('bbox')
    #             if fused_bbox is not None and list(fused_bbox) == list(det_bbox):
    #                 matched_fused.append(fd)
    #         if not matched_fused:
    #             continue
    #         extra = []
    #         extra.append(("--- Fused ---", color_text_title))
    #         for fd in matched_fused:
    #             method = fd.get('source', '')
    #             fused_conf = fd.get('person_confidence', 0.0)
    #             # Show fused confidence in overlay
    #             extra.append((f"Fused Confidence: {fused_conf:.3f}", (0, 0, 255)))
    #             extra.append((f"{method} Conf: {fused_conf:.3f}", color_text_body))
    #             # Fused positions
    #             geo_avg = fd.get('fused_geoposition_average')
    #             geo_bi = fd.get('fused_geoposition_bearing_intersection')
    #             if geo_avg:
    #                 flat = geo_avg.get('latitude', 0.0)
    #                 flon = geo_avg.get('longitude', 0.0)
    #                 extra.append((f"Avg Lat:{flat:.6f} Lon:{flon:.6f}", color_text_body))
    #             if geo_bi:
    #                 flat = geo_bi.get('latitude', 0.0)
    #                 flon = geo_bi.get('longitude', 0.0)
    #                 extra.append((f"Tri Lat:{flat:.6f} Lon:{flon:.6f}", color_text_body))
    #             # Fused distances (from this drone to each fused position)
    #             dist_avg = fd.get(f'distance_drone{drone_id}_average_m')
    #             dist_bi = fd.get(f'distance_drone{drone_id}_bearing_intersection_m')
    #             if dist_avg is not None:
    #                 extra.append((f"Avg Dist: {dist_avg:.1f}m", color_text_body))
    #             if dist_bi is not None:
    #                 extra.append((f"Tri Dist: {dist_bi:.1f}m", color_text_body))
    #             # Fused weapon info
    #             fused_has_weapon = fd.get('has_weapon', False)
    #             fused_weapon_conf = fd.get('weapon_confidence', 0.0)
    #             if fused_has_weapon:
    #                 extra.append((f"Fused ARMADO Conf: {fused_weapon_conf:.3f}", color_text_weapon))
    #         tracks_extra_lines[track_idx] = extra
    
    img_annotated = draw_bbox(
        image, tracks,
        show_confidence=show_confidence,
        tracks_extra_lines=tracks_extra_lines,
    )

    # Keep a simple top label for the visualization.
    cv2.putText(img_annotated, drone_label, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return img_annotated

def fused_visualization(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Make heights equal
    max_h = max(h1, h2)
    if h1 < max_h:
        img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if h2 < max_h:
        img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Concatenate horizontally
    combined = np.hstack([img1, img2])

    return combined