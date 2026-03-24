# import os
# import io
# import base64
# import time
# import json
# import requests
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as T
# from PIL import Image, ImageDraw, ImageFile
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# app = Flask(__name__)
# CORS(app)

# # ── Config ────────────────────────────────────────────────────────────
# IMG_SIZE   = 224
# DEVICE     = torch.device('cpu')  # Render free tier has no GPU
# MODEL_NAME = os.environ.get('MODEL_NAME', 'efficientnet')
# CKPT_PATH  = os.environ.get('CKPT_PATH', 'model/efficientnet_best.pth')
# CLASS_PATH = os.environ.get('CLASS_PATH', 'model/selected_classes_efficientnet.txt')

# # ── Load classes ──────────────────────────────────────────────────────
# with open(CLASS_PATH) as f:
#     selected = [l.strip() for l in f if l.strip()]
# print(f'Loaded {len(selected)} classes')

# # ── Load model ────────────────────────────────────────────────────────
# def build_model(name, num_classes):
#     if name == 'efficientnet':
#         m = models.efficientnet_b0(weights=None)
#         m.classifier = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(m.classifier[1].in_features, num_classes)
#         )
#     elif name == 'resnet50':
#         m = models.resnet50(weights=None)
#         m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
#     elif name == 'mobilenetv2':
#         m = models.mobilenet_v2(weights=None)
#         m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
#     elif name == 'mobilenetv3':
#         m = models.mobilenet_v3_small(weights=None)
#         m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
#     elif name == 'regnety':
#         m = models.regnet_y_400mf(weights=None)
#         m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
#     elif name == 'shufflenet':
#         m = models.shufflenet_v2_x1_0(weights=None)
#         m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
#     elif name == 'googlenet':
#         m = models.googlenet(weights=None, aux_logits=False)
#         m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
#     elif name == 'mnasnet':
#         m = models.mnasnet1_0(weights=None)
#         m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
#     else:
#         raise ValueError(f'Unknown model: {name}')
#     return m

# print(f'Loading model: {MODEL_NAME} from {CKPT_PATH}')
# model = build_model(MODEL_NAME, len(selected))
# model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
# model.eval()
# print('Model loaded.')

# # ── TTA transforms ────────────────────────────────────────────────────
# # tta_tfms = [
# #     T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
# #                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# #     T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(p=1.0),
# #                T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# #     T.Compose([T.Resize((int(IMG_SIZE*1.15),int(IMG_SIZE*1.15))), T.CenterCrop(IMG_SIZE),
# #                T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# #     T.Compose([T.Resize((int(IMG_SIZE*1.3),int(IMG_SIZE*1.3))), T.CenterCrop(IMG_SIZE),
# #                T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# #     T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),
# #                T.ColorJitter(brightness=0.15,contrast=0.15),
# #                T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# # ]

# tta_tfms = [
#     T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
#                T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
#     T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(p=1.0),
#                T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
# ]

# def tta_predict(crop):
#     acc = np.zeros(len(selected)); count = 0
#     with torch.no_grad():
#         for tfm in tta_tfms:
#             try:
#                 t = tfm(crop).unsqueeze(0).to(DEVICE)
#                 p = torch.softmax(model(t), 1).cpu().numpy()[0]
#                 acc += p; count += 1
#             except: pass
#     return acc / count if count else np.ones(len(selected)) / len(selected)

# # ── NMS ───────────────────────────────────────────────────────────────
# def iou(a, b):
#     ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
#     ix1,iy1 = max(ax1,bx1), max(ay1,by1)
#     ix2,iy2 = min(ax2,bx2), min(ay2,by2)
#     inter = max(0,ix2-ix1) * max(0,iy2-iy1)
#     union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
#     return inter/union if union > 0 else 0

# def nms(dets, iou_thresh=0.4):
#     if not dets: return []
#     dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
#     kept = []
#     while dets:
#         best = dets.pop(0); kept.append(best)
#         dets = [d for d in dets if iou(best['box'], d['box']) < iou_thresh]
#     return kept

# # def get_windows(W, H):
# #     wins = [(0, 0, W, H)]
# #     for scale in [0.5, 0.6, 0.75]:
# #         ww, wh = int(W*scale), int(H*scale)
# #         sx, sy = max(1, ww//2), max(1, wh//2)
# #         for y in range(0, H-wh+1, sy):
# #             for x in range(0, W-ww+1, sx):
# #                 wins.append((x, y, x+ww, y+wh))
# #     return wins


# def get_windows(W, H):
#     wins = [(0, 0, W, H)]
#     for scale in [0.6, 0.75]:  # removed 0.5
#         ww, wh = int(W*scale), int(H*scale)
#         sx, sy = max(1, ww//2), max(1, wh//2)
#         for y in range(0, H-wh+1, sy):
#             for x in range(0, W-ww+1, sx):
#                 wins.append((x, y, x+ww, y+wh))
#     return wins

# # ── Nutrition ─────────────────────────────────────────────────────────
# USDA_CACHE = {}
# QMAP = {
#     'aloo gobi':'potato cauliflower curry','aloo methi':'potato fenugreek curry',
#     'aloo mutter':'potato peas curry','aloo paratha':'stuffed potato flatbread',
#     'amritsari kulcha':'stuffed kulcha bread','anda curry':'egg curry Indian',
#     'balushahi':'balushahi Indian sweet','banana chips':'banana chips fried',
#     'besan laddu':'chickpea flour ladoo','bhindi masala':'okra masala Indian',
#     'biryani':'chicken biryani rice dish','boondi laddu':'boondi ladoo sweet',
#     'chaas':'buttermilk spiced Indian','chana masala':'chickpea curry masala',
#     'chapati':'whole wheat flatbread chapati','chicken pizza':'chicken pizza',
#     'chicken wings':'chicken wings','chikki':'peanut jaggery brittle',
#     'chivda':'beaten rice snack mix','chole bhature':'chickpea fried bread',
#     'dabeli':'spiced potato sandwich','dal khichdi':'lentil rice porridge',
#     'dhokla':'steamed chickpea cake','falooda':'rose milk vermicelli dessert',
#     'fish curry':'fish coconut curry Indian','gajar ka halwa':'carrot milk pudding',
#     'garlic bread':'garlic bread toast','garlic naan':'garlic naan flatbread',
#     'ghevar':'honeycomb sweet Rajasthan','grilled sandwich':'grilled vegetable sandwich',
#     'gujhia':'fried dumpling sweet','gulab jamun':'milk solid syrup balls',
#     'hara bhara kabab':'spinach peas patty','idiyappam':'steamed rice noodles',
#     'idli':'steamed rice lentil cake','jalebi':'deep fried sugar syrup',
#     'kaju katli':'cashew fudge barfi','khakhra':'crispy wheat flatbread',
#     'kheer':'rice milk pudding Indian','kulfi':'frozen milk dessert',
#     'margherita pizza':'margherita pizza','masala dosa':'crispy lentil crepe Indian',
#     'masala papad':'spiced lentil cracker','medu vada':'lentil donut fried',
#     'misal pav':'spicy sprouts curry bread','modak':'steamed sweet dumpling',
#     'moong dal halwa':'lentil flour pudding','murukku':'rice flour spiral snack',
#     'mysore pak':'chickpea flour fudge','navratan korma':'mixed vegetable cream curry',
#     'neer dosa':'thin rice crepe','onion pakoda':'onion batter fritters',
#     'palak paneer':'spinach cottage cheese curry','paneer masala':'cottage cheese tomato curry',
#     'paneer pizza':'cottage cheese pizza','pani puri':'hollow crispy puri',
#     'paniyaram':'rice lentil dumplings','papdi chaat':'crispy wafer chaat',
#     'patrode':'taro colocasia rolls','pav bhaji':'spiced mashed vegetables bread',
#     'pepperoni pizza':'pepperoni pizza','phirni':'ground rice milk pudding',
#     'poha':'flattened rice breakfast','pongal':'lentil rice porridge South Indian',
#     'puri bhaji':'deep fried bread potato curry','rajma chawal':'kidney bean curry rice',
#     'rasgulla':'cottage cheese syrup balls','rava dosa':'semolina lacy crepe',
#     'sabudana khichdi':'tapioca pearl pilaf','sabudana vada':'tapioca potato patty',
#     'samosa':'triangular fried pastry Indian','seekh kebab':'minced meat skewer',
#     'set dosa':'soft thick pancake','sev puri':'crispy flat chaat',
#     'solkadhi':'kokum coconut curry drink','steamed momo':'steamed dumplings',
#     'thali':'Indian complete meal plate','thukpa':'Tibetan noodle broth soup',
#     'uttapam':'thick rice lentil pancake','vada pav':'potato fritter bread',
# }

# def get_nutrition(food, portion_g=150):
#     key = food.lower()
#     if key in USDA_CACHE:
#         c = USDA_CACHE[key]; s = portion_g / 100
#         return {'calories':round(c['cal']*s), 'protein':round(c['prot']*s,1),
#                 'carbs':round(c['carb']*s,1), 'fat':round(c['fat']*s,1),
#                 'portion_g':portion_g, 'source':c['src']}
#     query = QMAP.get(key, food)
#     try:
#         r = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search',
#                          params={'query':query,'api_key':'DEMO_KEY','pageSize':3},
#                          timeout=8)
#         foods = r.json().get('foods', [])
#         if foods:
#             fd = foods[0]
#             n  = {x['nutrientName']:x['value'] for x in fd.get('foodNutrients',[])}
#             c  = {
#                 'cal':  n.get('Energy', n.get('Energy (Atwater General Factors)', 200)),
#                 'prot': n.get('Protein', 5.0),
#                 'carb': n.get('Carbohydrate, by difference', 30.0),
#                 'fat':  n.get('Total lipid (fat)', 7.0),
#                 'src':  f"USDA FDC: {fd.get('description', query)}"
#             }
#             USDA_CACHE[key] = c; s = portion_g / 100
#             return {'calories':round(c['cal']*s), 'protein':round(c['prot']*s,1),
#                     'carbs':round(c['carb']*s,1), 'fat':round(c['fat']*s,1),
#                     'portion_g':portion_g, 'source':c['src']}
#     except: pass
#     s = portion_g / 100
#     return {'calories':round(200*s), 'protein':round(5*s,1), 'carbs':round(30*s,1),
#             'fat':round(7*s,1), 'portion_g':portion_g, 'source':'Fallback'}

# # ── Detection ─────────────────────────────────────────────────────────
# COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
#           '#1abc9c','#e67e22','#e91e63','#00bcd4','#8e44ad']

# def detect(pil_image, conf_thresh=0.55):
#     W, H = pil_image.size
#     raw = []
#     for box in get_windows(W, H):
#         probs  = tta_predict(pil_image.crop(box))
#         top_i  = int(np.argmax(probs))
#         conf   = float(probs[top_i])
#         if conf >= conf_thresh:
#             top3_i = np.argsort(probs)[-3:][::-1]
#             raw.append({'box':box, 'food':selected[top_i], 'conf':conf,
#                         'top3':[(selected[i], round(float(probs[i])*100,1)) for i in top3_i]})
#     kept = nms(raw, 0.4)
#     if not kept:
#         probs  = tta_predict(pil_image)
#         top_i  = int(np.argmax(probs))
#         top3_i = np.argsort(probs)[-3:][::-1]
#         kept   = [{'box':(0,0,W,H), 'food':selected[top_i], 'conf':float(probs[top_i]),
#                    'top3':[(selected[i], round(float(probs[i])*100,1)) for i in top3_i]}]

#     ann  = pil_image.copy()
#     draw = ImageDraw.Draw(ann)
#     results = []
#     for i, d in enumerate(kept):
#         x1,y1,x2,y2 = d['box']
#         ratio     = ((x2-x1)*(y2-y1)) / (W*H)
#         portion_g = max(50, min(400, int(300*ratio*4)))
#         nut       = get_nutrition(d['food'], portion_g)
#         col       = COLORS[i % len(COLORS)]
#         for t in range(4): draw.rectangle([x1-t,y1-t,x2+t,y2+t], outline=col)
#         l1 = f"{i+1}. {d['food'].title()} {d['conf']*100:.0f}%"
#         l2 = f"{nut['calories']} kcal"
#         ly = max(0, y1-48); bw = max(len(l1),len(l2))*7+14
#         draw.rectangle([x1,ly,x1+bw,ly+46], fill=col)
#         draw.text((x1+5,ly+4),  l1, fill='white')
#         draw.text((x1+5,ly+26), l2, fill='white')
#         results.append({
#             'id':           i+1,
#             'food':         d['food'],
#             'food_display': d['food'].replace('_',' ').title(),
#             'confidence':   round(d['conf']*100, 1),
#             'top3':         d['top3'],
#             'box':          list(d['box']),
#             'color':        col,
#             'nutrition':    nut
#         })

#     buf = io.BytesIO()
#     ann.save(buf, format='JPEG', quality=90)
#     return results, base64.b64encode(buf.getvalue()).decode()

# # ── Routes ────────────────────────────────────────────────────────────
# @app.route('/', methods=['GET'])
# def index():
#     return jsonify({'name':'NutriLens API','model':MODEL_NAME,'classes':len(selected)})

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status':'ok','model':MODEL_NAME,'classes':len(selected)})

# @app.route('/classes', methods=['GET'])
# def get_classes():
#     return jsonify({'classes':selected})

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         if not data or 'image' not in data:
#             return jsonify({'error':'No image provided'}), 400

#         img_b64 = data['image']
#         if ',' in img_b64: img_b64 = img_b64.split(',')[1]
#         pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')

#         # Resize large images for speed on free tier
#         MAX_DIM = 700
#         if max(pil_img.size) > MAX_DIM:
#             ratio   = MAX_DIM / max(pil_img.size)
#             pil_img = pil_img.resize(
#                 (int(pil_img.width*ratio), int(pil_img.height*ratio)),
#                 Image.LANCZOS
#             )

#         conf_thresh = float(data.get('conf_thresh', 0.55))
#         results, annotated_b64 = detect(pil_img, conf_thresh)

#         return jsonify({
#             'success':    True,
#             'detections': results,
#             'annotated':  f'data:image/jpeg;base64,{annotated_b64}',
#             'total_nutrition': {
#                 'calories': sum(r['nutrition']['calories'] for r in results),
#                 'protein':  sum(r['nutrition']['protein']  for r in results),
#                 'carbs':    sum(r['nutrition']['carbs']    for r in results),
#                 'fat':      sum(r['nutrition']['fat']      for r in results),
#             },
#             'model':      MODEL_NAME,
#             'image_size': list(pil_img.size)
#         })

#     except Exception as e:
#         import traceback
#         return jsonify({'error':str(e),'trace':traceback.format_exc()}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)








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
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/predict', methods=['OPTIONS'])
@app.route('/health', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def options_handler():
    response = jsonify({'status': 'ok'})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response, 200

# ── Config ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
DEVICE     = torch.device('cpu')
MODEL_NAME = os.environ.get('MODEL_NAME', 'efficientnet')
CKPT_PATH  = os.environ.get('CKPT_PATH', 'model/efficientnet_best.pth')
CLASS_PATH = os.environ.get('CLASS_PATH', 'model/selected_classes_efficientnet.txt')
MAX_DIM    = 480   # smaller = faster, still accurate enough

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
# Torch compile for faster CPU inference (PyTorch 2.x)
try:
    model = torch.compile(model, mode='reduce-overhead')
    print('Model compiled with torch.compile')
except Exception:
    print('torch.compile unavailable, using standard model')
print('Model ready.')

# ── Single transform (no TTA — speed priority) ────────────────────────
tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def batch_predict(crops):
    """Run all crops through the model in ONE forward pass."""
    if not crops:
        return np.array([])
    batch = torch.stack([tfm(c) for c in crops]).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        # Temperature scaling = 1.5 gives better calibrated confidence
        probs  = torch.softmax(logits / 1.5, dim=1).cpu().numpy()
    return probs

# ── Smart crop strategy ───────────────────────────────────────────────
def get_crops(img):
    """
    Returns list of (crop_pil, box) tuples.
    Strategy:
      1. Full image always included
      2. If image is wide/tall enough: 4 quadrant crops
      3. 2 horizontal half-crops (top/bottom) for dishes laid side by side
    Total = 7 crops max, all run in ONE batched forward pass.
    """
    W, H = img.size
    crops_and_boxes = [(img, (0, 0, W, H))]

    # Only add sub-crops if image is large enough to be meaningful
    if W >= 200 and H >= 200:
        hw, hh = W // 2, H // 2
        quads = [
            (0,  0,  hw, hh),   # top-left
            (hw, 0,  W,  hh),   # top-right
            (0,  hh, hw, H),    # bottom-left
            (hw, hh, W,  H),    # bottom-right
        ]
        for box in quads:
            crops_and_boxes.append((img.crop(box), box))

        # Horizontal halves (good for side-by-side dishes)
        crops_and_boxes.append((img.crop((0, 0, W, hh)), (0, 0, W, hh)))
        crops_and_boxes.append((img.crop((0, hh, W, H)), (0, hh, W, H)))

    return crops_and_boxes

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

# ── Local nutrition table (per 100g) — no API call needed ────────────
# Format: cal, protein, carbs, fat
NUTRITION_DB = {
    'aloo paratha':     (200, 4.5, 28.0, 8.0),
    'anda curry':       (165, 11.0, 6.0, 11.0),
    'biryani':          (180, 8.0, 25.0, 5.5),
    'chana masala':     (150, 7.5, 20.0, 4.5),
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
FALLBACK_NUTRITION = (200, 5.0, 30.0, 7.0)

def get_nutrition(food, portion_g=150):
    key   = food.lower().strip()
    base  = NUTRITION_DB.get(key, FALLBACK_NUTRITION)
    s     = portion_g / 100
    src   = 'NutriLens DB' if key in NUTRITION_DB else 'Estimated'
    return {
        'calories':  round(base[0] * s),
        'protein':   round(base[1] * s, 1),
        'carbs':     round(base[2] * s, 1),
        'fat':       round(base[3] * s, 1),
        'portion_g': portion_g,
        'source':    src
    }

# ── Colors ────────────────────────────────────────────────────────────
COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#e91e63','#00bcd4','#8e44ad']

# ── Main detect function ──────────────────────────────────────────────
def detect(pil_image, conf_thresh=0.55):
    W, H = pil_image.size

    # Get all crops and run ONE batched forward pass
    crops_and_boxes = get_crops(pil_image)
    crops  = [c for c, _ in crops_and_boxes]
    boxes  = [b for _, b in crops_and_boxes]
    probs  = batch_predict(crops)   # shape: (N, num_classes)

    # Filter by confidence threshold
    raw = []
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        top_i  = int(np.argmax(prob))
        conf   = float(prob[top_i])
        if conf >= conf_thresh:
            top3_i = np.argsort(prob)[-3:][::-1]
            raw.append({
                'box':  box,
                'food': selected[top_i],
                'conf': conf,
                'top3': [(selected[j], round(float(prob[j])*100, 1)) for j in top3_i]
            })

    kept = nms(raw, iou_thresh=0.45)

    # If nothing passed threshold, return best single guess from full image
    if not kept:
        prob   = probs[0]   # full image is always index 0
        top_i  = int(np.argmax(prob))
        top3_i = np.argsort(prob)[-3:][::-1]
        kept   = [{
            'box':  (0, 0, W, H),
            'food': selected[top_i],
            'conf': float(prob[top_i]),
            'top3': [(selected[j], round(float(prob[j])*100, 1)) for j in top3_i]
        }]

    # Annotate image
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
        area      = (x2-x1) * (y2-y1)
        ratio     = area / (W * H)
        portion_g = max(50, min(400, int(300 * ratio * 4)))
        nut       = get_nutrition(d['food'], portion_g)
        col       = COLORS[i % len(COLORS)]

        # Draw box
        for t in range(3):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=col)

        # Draw label
        l1 = f"{i+1}. {d['food'].replace('_',' ').title()}  {d['conf']*100:.0f}%"
        l2 = f"{nut['calories']} kcal  |  {nut['protein']}g P  {nut['carbs']}g C  {nut['fat']}g F"
        ly = max(0, y1 - 50)
        bw = max(len(l1), len(l2)) * 7 + 14
        draw.rectangle([x1, ly, x1+bw, ly+48], fill=col)
        draw.text((x1+6, ly+4),  l1, fill='white', font=font_big)
        draw.text((x1+6, ly+26), l2, fill='white', font=font_small)

        results.append({
            'id':           i + 1,
            'food':         d['food'],
            'food_display': d['food'].replace('_', ' ').title(),
            'confidence':   round(d['conf'] * 100, 1),
            'top3':         d['top3'],
            'box':          list(d['box']),
            'color':        col,
            'nutrition':    nut
        })

    buf = io.BytesIO()
    ann.save(buf, format='JPEG', quality=88)
    return results, base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({'name': 'NutriLens API', 'model': MODEL_NAME, 'classes': NUM_CLASSES})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': MODEL_NAME, 'classes': NUM_CLASSES})

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': selected})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')

        # Resize for speed
        if max(pil_img.size) > MAX_DIM:
            ratio   = MAX_DIM / max(pil_img.size)
            pil_img = pil_img.resize(
                (int(pil_img.width * ratio), int(pil_img.height * ratio)),
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
