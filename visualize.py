import json
from PIL import Image, ImageDraw

with open('data/hw02_preds/preds_train.json') as f:
    preds_train = json.load(f)
with open('data/hw02_preds/preds_test.json') as f:
    preds_test = json.load(f)

for filename in preds_train:
    I = Image.open("data/RedLights2011_Medium/" + filename)
    draw = ImageDraw.Draw(I)
    for box in preds_train[filename]:
        draw.rectangle([box[1], box[0], box[3], box[2]])
    I.save("data/hw02_visualizations/" + filename)
for filename in preds_test:
    I = Image.open("data/RedLights2011_Medium/" + filename)
    draw = ImageDraw.Draw(I)
    for box in preds_test[filename]:
        draw.rectangle([box[1], box[0], box[3], box[2]])
    I.save("data/hw02_visualizations/" + filename)

with open('data/hw02_annotations/annotations.json') as f:
    gts = json.load(f)
for filename in gts:
    I = Image.open("data/RedLights2011_Medium/" + filename)
    draw = ImageDraw.Draw(I)
    for box in gts[filename]:
        box = [int(i) for i in box]
        draw.rectangle([box[1], box[0], box[3], box[2]])
    I.save("data/hw02_gt/" + filename)


