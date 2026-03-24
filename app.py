import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageFile
from flask import Flask, request, jsonify
from flask_cors import CORS

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

@app.after_request
def after_request(response):
    response.headers.set('Access-Control-Allow-Origin', '*')
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# ── Config ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
DEVICE     = torch.device('cpu')
MODEL_NAME = os.environ.get('MODEL_NAME', 'efficientnet')
CKPT_PATH  = os.environ.get('CKPT_PATH', 'model/efficientnet_best.pth')
CLASS_PATH = os.environ.get('CLASS_PATH', 'model/selected_classes_efficientnet.txt')
MAX_DIM    = 380

# ── Load classes ──────────────────────────────────────────────────────
with open(CLASS_PATH) as f:
    selected = [l.strip() for l in f if l.strip()]
NUM_CLASSES = len(selected)
print(f'Loaded {NUM_CLASSES} classes')

# ── Build & load model ────────────────────────────────────────────────
def build_model(name, num_classes):
    if name == 'efficientnet':
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[1].in_features, num_classes)
        )
    elif name == 'resnet50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif name == 'mobilenetv2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == 'mobilenetv3':
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == 'regnety':
        m = models.regnet_y_400mf(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif name == 'shufflenet':
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif name == 'googlenet':
        m = models.googlenet(weights=None, aux_logits=False)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    elif name == 'mnasnet':
        m = models.mnasnet1_0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')
    return m

print(f'Loading model: {MODEL_NAME} from {CKPT_PATH}')
model = build_model(MODEL_NAME, NUM_CLASSES)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print('Model ready.')

# ── Transform ─────────────────────────────────────────────────────────
tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def batch_predict(crops):
    if not crops:
        return np.array([])
    batch = torch.stack([tfm(c) for c in crops]).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(batch) / 1.5, dim=1).cpu().numpy()
    return probs

# ── Smart crops ───────────────────────────────────────────────────────
def get_crops(img):
    # Single full image only — fastest on free tier CPU
    W, H = img.size
    return [(img, (0, 0, W, H))]

# ── NMS ───────────────────────────────────────────────────────────────
def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union > 0 else 0

def nms(dets, iou_thresh=0.45):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0)
        kept.append(best)
        dets = [d for d in dets if iou(best['box'], d['box']) < iou_thresh]
    return kept

# ── Local nutrition DB (per 100g): cal, protein, carbs, fat ──────────
NUTRITION_DB = {
    'aloo paratha':     (200, 4.5, 28.0, 8.0),
    'anda curry':       (165, 11.0, 6.0,  11.0),
    'biryani':          (180, 8.0,  25.0, 5.5),
    'chana masala':     (150, 7.5,  20.0, 4.5),
    'chicken pizza':    (270, 13.0, 30.0, 11.0),
    'chicken wings':    (290, 24.0, 0.0,  21.0),
    'garlic bread':     (310, 7.0,  45.0, 11.0),
    'garlic naan':      (300, 8.0,  50.0, 8.0),
    'grilled sandwich': (220, 9.0,  28.0, 8.0),
    'gulab jamun':      (380, 5.0,  55.0, 15.0),
    'idli':             (130, 4.0,  25.0, 0.5),
    'kulfi':            (220, 5.0,  28.0, 10.0),
    'masala dosa':      (165, 4.0,  28.0, 4.5),
    'palak paneer':     (180, 8.0,  8.0,  13.0),
    'paneer masala':    (200, 9.0,  10.0, 14.0),
    'pav bhaji':        (175, 4.5,  28.0, 5.0),
    'samosa':           (310, 5.0,  35.0, 17.0),
    'seekh kebab':      (250, 18.0, 5.0,  18.0),
    'uttapam':          (150, 4.5,  26.0, 3.0),
    'vada pav':         (290, 6.5,  42.0, 10.0),
}
FALLBACK = (200, 5.0, 30.0, 7.0)

def get_nutrition(food, portion_g=150):
    key  = food.lower().strip()
    base = NUTRITION_DB.get(key, FALLBACK)
    s    = portion_g / 100
    return {
        'calories':  round(base[0] * s),
        'protein':   round(base[1] * s, 1),
        'carbs':     round(base[2] * s, 1),
        'fat':       round(base[3] * s, 1),
        'portion_g': portion_g,
        'source':    'NutriLens DB' if key in NUTRITION_DB else 'Estimated'
    }

# ── Colors ────────────────────────────────────────────────────────────
COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#e91e63','#00bcd4','#8e44ad']

# ── Detect ────────────────────────────────────────────────────────────
def detect(pil_image, conf_thresh=0.55):
    W, H = pil_image.size
    crops_and_boxes = get_crops(pil_image)
    crops = [c for c, _ in crops_and_boxes]
    boxes = [b for _, b in crops_and_boxes]
    probs = batch_predict(crops)

    raw = []
    for box, prob in zip(boxes, probs):
        top_i = int(np.argmax(prob))
        conf  = float(prob[top_i])
        if conf >= conf_thresh:
            top3_i = np.argsort(prob)[-3:][::-1]
            raw.append({
                'box':  box,
                'food': selected[top_i],
                'conf': conf,
                'top3': [(selected[j], round(float(prob[j])*100, 1)) for j in top3_i]
            })

    kept = nms(raw, iou_thresh=0.45)

    if not kept:
        prob   = probs[0]
        top_i  = int(np.argmax(prob))
        top3_i = np.argsort(prob)[-3:][::-1]
        kept   = [{
            'box':  (0, 0, W, H),
            'food': selected[top_i],
            'conf': float(prob[top_i]),
            'top3': [(selected[j], round(float(prob[j])*100, 1)) for j in top3_i]
        }]

    ann  = pil_image.copy()
    draw = ImageDraw.Draw(ann)
    try:
        font_big   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 15)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except Exception:
        font_big = font_small = ImageFont.load_default()

    results = []
    for i, d in enumerate(kept):
        x1, y1, x2, y2 = d['box']
        ratio     = ((x2-x1)*(y2-y1)) / (W*H)
        portion_g = max(50, min(400, int(300*ratio*4)))
        nut       = get_nutrition(d['food'], portion_g)
        col       = COLORS[i % len(COLORS)]
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=col)
        l1 = f"{i+1}. {d['food'].replace('_',' ').title()}  {d['conf']*100:.0f}%"
        l2 = f"{nut['calories']} kcal | {nut['protein']}g P  {nut['carbs']}g C  {nut['fat']}g F"
        ly = max(0, y1-50)
        bw = max(len(l1), len(l2)) * 7 + 14
        draw.rectangle([x1, ly, x1+bw, ly+48], fill=col)
        draw.text((x1+6, ly+4),  l1, fill='white', font=font_big)
        draw.text((x1+6, ly+26), l2, fill='white', font=font_small)
        results.append({
            'id':           i+1,
            'food':         d['food'],
            'food_display': d['food'].replace('_', ' ').title(),
            'confidence':   round(d['conf']*100, 1),
            'top3':         d['top3'],
            'box':          list(d['box']),
            'color':        col,
            'nutrition':    nut
        })

    buf = io.BytesIO()
    ann.save(buf, format='JPEG', quality=88)
    return results, base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    return jsonify({'name': 'NutriLens API', 'model': MODEL_NAME, 'classes': NUM_CLASSES})

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL_NAME, 'classes': NUM_CLASSES})

@app.route('/classes', methods=['GET', 'OPTIONS'])
def get_classes():
    return jsonify({'classes': selected})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')

        if max(pil_img.size) > MAX_DIM:
            ratio   = MAX_DIM / max(pil_img.size)
            pil_img = pil_img.resize(
                (int(pil_img.width*ratio), int(pil_img.height*ratio)),
                Image.LANCZOS
            )

        conf_thresh = float(data.get('conf_thresh', 0.55))
        results, annotated_b64 = detect(pil_img, conf_thresh)

        return jsonify({
            'success':    True,
            'detections': results,
            'annotated':  f'data:image/jpeg;base64,{annotated_b64}',
            'total_nutrition': {
                'calories': sum(r['nutrition']['calories'] for r in results),
                'protein':  round(sum(r['nutrition']['protein']  for r in results), 1),
                'carbs':    round(sum(r['nutrition']['carbs']    for r in results), 1),
                'fat':      round(sum(r['nutrition']['fat']      for r in results), 1),
            },
            'model':      MODEL_NAME,
            'image_size': list(pil_img.size)
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
