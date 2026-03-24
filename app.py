import os
import io
import base64
import time
import json
import requests
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFile
from flask import Flask, request, jsonify
from flask_cors import CORS

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────
IMG_SIZE   = 224
DEVICE     = torch.device('cpu')  # Render free tier has no GPU
MODEL_NAME = os.environ.get('MODEL_NAME', 'efficientnet')
CKPT_PATH  = os.environ.get('CKPT_PATH', 'model/efficientnet_best.pth')
CLASS_PATH = os.environ.get('CLASS_PATH', 'model/selected_classes.txt')

# ── Load classes ──────────────────────────────────────────────────────
with open(CLASS_PATH) as f:
    selected = [l.strip() for l in f if l.strip()]
print(f'Loaded {len(selected)} classes')

# ── Load model ────────────────────────────────────────────────────────
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
model = build_model(MODEL_NAME, len(selected))
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print('Model loaded.')

# ── TTA transforms ────────────────────────────────────────────────────
tta_tfms = [
    T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.ToTensor(),
               T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)), T.RandomHorizontalFlip(p=1.0),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((int(IMG_SIZE*1.15),int(IMG_SIZE*1.15))), T.CenterCrop(IMG_SIZE),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((int(IMG_SIZE*1.3),int(IMG_SIZE*1.3))), T.CenterCrop(IMG_SIZE),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),
               T.ColorJitter(brightness=0.15,contrast=0.15),
               T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]

def tta_predict(crop):
    acc = np.zeros(len(selected)); count = 0
    with torch.no_grad():
        for tfm in tta_tfms:
            try:
                t = tfm(crop).unsqueeze(0).to(DEVICE)
                p = torch.softmax(model(t), 1).cpu().numpy()[0]
                acc += p; count += 1
            except: pass
    return acc / count if count else np.ones(len(selected)) / len(selected)

# ── NMS ───────────────────────────────────────────────────────────────
def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0

def nms(dets, iou_thresh=0.4):
    if not dets: return []
    dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0); kept.append(best)
        dets = [d for d in dets if iou(best['box'], d['box']) < iou_thresh]
    return kept

def get_windows(W, H):
    wins = [(0, 0, W, H)]
    for scale in [0.5, 0.6, 0.75]:
        ww, wh = int(W*scale), int(H*scale)
        sx, sy = max(1, ww//2), max(1, wh//2)
        for y in range(0, H-wh+1, sy):
            for x in range(0, W-ww+1, sx):
                wins.append((x, y, x+ww, y+wh))
    return wins

# ── Nutrition ─────────────────────────────────────────────────────────
USDA_CACHE = {}
QMAP = {
    'aloo gobi':'potato cauliflower curry','aloo methi':'potato fenugreek curry',
    'aloo mutter':'potato peas curry','aloo paratha':'stuffed potato flatbread',
    'amritsari kulcha':'stuffed kulcha bread','anda curry':'egg curry Indian',
    'balushahi':'balushahi Indian sweet','banana chips':'banana chips fried',
    'besan laddu':'chickpea flour ladoo','bhindi masala':'okra masala Indian',
    'biryani':'chicken biryani rice dish','boondi laddu':'boondi ladoo sweet',
    'chaas':'buttermilk spiced Indian','chana masala':'chickpea curry masala',
    'chapati':'whole wheat flatbread chapati','chicken pizza':'chicken pizza',
    'chicken wings':'chicken wings','chikki':'peanut jaggery brittle',
    'chivda':'beaten rice snack mix','chole bhature':'chickpea fried bread',
    'dabeli':'spiced potato sandwich','dal khichdi':'lentil rice porridge',
    'dhokla':'steamed chickpea cake','falooda':'rose milk vermicelli dessert',
    'fish curry':'fish coconut curry Indian','gajar ka halwa':'carrot milk pudding',
    'garlic bread':'garlic bread toast','garlic naan':'garlic naan flatbread',
    'ghevar':'honeycomb sweet Rajasthan','grilled sandwich':'grilled vegetable sandwich',
    'gujhia':'fried dumpling sweet','gulab jamun':'milk solid syrup balls',
    'hara bhara kabab':'spinach peas patty','idiyappam':'steamed rice noodles',
    'idli':'steamed rice lentil cake','jalebi':'deep fried sugar syrup',
    'kaju katli':'cashew fudge barfi','khakhra':'crispy wheat flatbread',
    'kheer':'rice milk pudding Indian','kulfi':'frozen milk dessert',
    'margherita pizza':'margherita pizza','masala dosa':'crispy lentil crepe Indian',
    'masala papad':'spiced lentil cracker','medu vada':'lentil donut fried',
    'misal pav':'spicy sprouts curry bread','modak':'steamed sweet dumpling',
    'moong dal halwa':'lentil flour pudding','murukku':'rice flour spiral snack',
    'mysore pak':'chickpea flour fudge','navratan korma':'mixed vegetable cream curry',
    'neer dosa':'thin rice crepe','onion pakoda':'onion batter fritters',
    'palak paneer':'spinach cottage cheese curry','paneer masala':'cottage cheese tomato curry',
    'paneer pizza':'cottage cheese pizza','pani puri':'hollow crispy puri',
    'paniyaram':'rice lentil dumplings','papdi chaat':'crispy wafer chaat',
    'patrode':'taro colocasia rolls','pav bhaji':'spiced mashed vegetables bread',
    'pepperoni pizza':'pepperoni pizza','phirni':'ground rice milk pudding',
    'poha':'flattened rice breakfast','pongal':'lentil rice porridge South Indian',
    'puri bhaji':'deep fried bread potato curry','rajma chawal':'kidney bean curry rice',
    'rasgulla':'cottage cheese syrup balls','rava dosa':'semolina lacy crepe',
    'sabudana khichdi':'tapioca pearl pilaf','sabudana vada':'tapioca potato patty',
    'samosa':'triangular fried pastry Indian','seekh kebab':'minced meat skewer',
    'set dosa':'soft thick pancake','sev puri':'crispy flat chaat',
    'solkadhi':'kokum coconut curry drink','steamed momo':'steamed dumplings',
    'thali':'Indian complete meal plate','thukpa':'Tibetan noodle broth soup',
    'uttapam':'thick rice lentil pancake','vada pav':'potato fritter bread',
}

def get_nutrition(food, portion_g=150):
    key = food.lower()
    if key in USDA_CACHE:
        c = USDA_CACHE[key]; s = portion_g / 100
        return {'calories':round(c['cal']*s), 'protein':round(c['prot']*s,1),
                'carbs':round(c['carb']*s,1), 'fat':round(c['fat']*s,1),
                'portion_g':portion_g, 'source':c['src']}
    query = QMAP.get(key, food)
    try:
        r = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search',
                         params={'query':query,'api_key':'DEMO_KEY','pageSize':3},
                         timeout=8)
        foods = r.json().get('foods', [])
        if foods:
            fd = foods[0]
            n  = {x['nutrientName']:x['value'] for x in fd.get('foodNutrients',[])}
            c  = {
                'cal':  n.get('Energy', n.get('Energy (Atwater General Factors)', 200)),
                'prot': n.get('Protein', 5.0),
                'carb': n.get('Carbohydrate, by difference', 30.0),
                'fat':  n.get('Total lipid (fat)', 7.0),
                'src':  f"USDA FDC: {fd.get('description', query)}"
            }
            USDA_CACHE[key] = c; s = portion_g / 100
            return {'calories':round(c['cal']*s), 'protein':round(c['prot']*s,1),
                    'carbs':round(c['carb']*s,1), 'fat':round(c['fat']*s,1),
                    'portion_g':portion_g, 'source':c['src']}
    except: pass
    s = portion_g / 100
    return {'calories':round(200*s), 'protein':round(5*s,1), 'carbs':round(30*s,1),
            'fat':round(7*s,1), 'portion_g':portion_g, 'source':'Fallback'}

# ── Detection ─────────────────────────────────────────────────────────
COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#e91e63','#00bcd4','#8e44ad']

def detect(pil_image, conf_thresh=0.55):
    W, H = pil_image.size
    raw = []
    for box in get_windows(W, H):
        probs  = tta_predict(pil_image.crop(box))
        top_i  = int(np.argmax(probs))
        conf   = float(probs[top_i])
        if conf >= conf_thresh:
            top3_i = np.argsort(probs)[-3:][::-1]
            raw.append({'box':box, 'food':selected[top_i], 'conf':conf,
                        'top3':[(selected[i], round(float(probs[i])*100,1)) for i in top3_i]})
    kept = nms(raw, 0.4)
    if not kept:
        probs  = tta_predict(pil_image)
        top_i  = int(np.argmax(probs))
        top3_i = np.argsort(probs)[-3:][::-1]
        kept   = [{'box':(0,0,W,H), 'food':selected[top_i], 'conf':float(probs[top_i]),
                   'top3':[(selected[i], round(float(probs[i])*100,1)) for i in top3_i]}]

    ann  = pil_image.copy()
    draw = ImageDraw.Draw(ann)
    results = []
    for i, d in enumerate(kept):
        x1,y1,x2,y2 = d['box']
        ratio     = ((x2-x1)*(y2-y1)) / (W*H)
        portion_g = max(50, min(400, int(300*ratio*4)))
        nut       = get_nutrition(d['food'], portion_g)
        col       = COLORS[i % len(COLORS)]
        for t in range(4): draw.rectangle([x1-t,y1-t,x2+t,y2+t], outline=col)
        l1 = f"{i+1}. {d['food'].title()} {d['conf']*100:.0f}%"
        l2 = f"{nut['calories']} kcal"
        ly = max(0, y1-48); bw = max(len(l1),len(l2))*7+14
        draw.rectangle([x1,ly,x1+bw,ly+46], fill=col)
        draw.text((x1+5,ly+4),  l1, fill='white')
        draw.text((x1+5,ly+26), l2, fill='white')
        results.append({
            'id':           i+1,
            'food':         d['food'],
            'food_display': d['food'].replace('_',' ').title(),
            'confidence':   round(d['conf']*100, 1),
            'top3':         d['top3'],
            'box':          list(d['box']),
            'color':        col,
            'nutrition':    nut
        })

    buf = io.BytesIO()
    ann.save(buf, format='JPEG', quality=90)
    return results, base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({'name':'NutriLens API','model':MODEL_NAME,'classes':len(selected)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','model':MODEL_NAME,'classes':len(selected)})

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes':selected})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error':'No image provided'}), 400

        img_b64 = data['image']
        if ',' in img_b64: img_b64 = img_b64.split(',')[1]
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')

        # Resize large images for speed on free tier
        MAX_DIM = 700
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
                'protein':  sum(r['nutrition']['protein']  for r in results),
                'carbs':    sum(r['nutrition']['carbs']    for r in results),
                'fat':      sum(r['nutrition']['fat']      for r in results),
            },
            'model':      MODEL_NAME,
            'image_size': list(pil_img.size)
        })

    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
