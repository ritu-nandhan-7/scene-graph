from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from scene_graph.mapping import build_yolo_vocabulary
out = root / 'experiments' / 'browser-feasibility' / 'browser-yolo-test' / 'public' / 'references'
model = YOLOWorld(str(root / 'notebook' / 'models' / 'yolov8s-world.pt'))
model.set_classes(build_yolo_vocabulary(json.loads((root / 'datasets' / 'visual_genome' / 'VG-SGG-dicts.json').read_text(encoding='utf-8'))))
model.model.eval()
for path in (root / 'datasets' / 'visual_genome' / 'VG_100K_2' / '1.jpg', root / 'datasets' / 'visual_genome' / 'VG_100K_2' / '100.jpg', root / 'datasets' / 'visual_genome' / 'VG_100K' / '107899.jpg'):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    model.predict(str(path), conf=.25, imgsz=640, rect=False, device='cpu', verbose=False)
    tensor = model.predictor.preprocess([bgr]).detach().cpu().numpy()
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    np.save(out / f'input-pytorch-{path.stem}.npy', tensor)
    print(path.name, tensor.shape, tensor.min(), tensor.max())
