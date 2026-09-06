import csv, yaml, torch, numpy as np
from dover.datasets import ViewDecompositionDataset
from dover.models import DOVER

opt = yaml.safe_load(open('divide_repro.yml', encoding='utf-8'))
dopt = opt['data']['val-dividemaxwell']['args']
ds = ViewDecompositionDataset(dopt)
idx = [i for i, v in enumerate(ds.video_infos) if v['filename'].endswith('2284.mp4')][0]
data = ds[idx]
assert len(data.keys()) > 1, 'decode failed'

model = DOVER(**opt['model']['args']).cuda().eval()
model.load_state_dict(torch.load('pretrained_weights/DOVER.pth', map_location='cuda'))

video = {}
for key in ['aesthetic', 'technical']:
    v = data[key].cuda().unsqueeze(0)
    b, c, t, h, w = v.shape
    nc = data['num_clips'][key]
    video[key] = v.reshape(b, c, nc, t//nc, h, w).permute(0,2,1,3,4,5).reshape(b*nc, c, t//nc, h, w)

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    results = model(video, reduce_scores=False)
    results = [np.mean(x.float().cpu().numpy()) for x in results]

t, a = (results[1] - 0.1107) / 0.07355, (results[0] + 0.08285) / 0.03774
x = t * 0.6104 + a * 0.3896
a_s, t_s, o_s = 1/(1+np.exp(-a)), 1/(1+np.exp(-t)), 1/(1+np.exp(-x))
print('2284.mp4 scores:', a_s, t_s, o_s)

with open('zero_shot_predictions.csv', 'a', newline='', encoding='utf-8') as f:
    csv.writer(f).writerow(['2284.mp4', a_s, t_s, o_s, '', ''])
print('appended')
