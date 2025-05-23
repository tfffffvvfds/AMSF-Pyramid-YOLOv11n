import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

# 定义默认路径常量
DEFAULT_ANNO_JSON = r'D:\Users\hp\Desktop\dataSet\KY004：无人机路面破损检测数据集UAV-PDD2023\dataJson\dataTest.json'
DEFAULT_PRED_JSON = r'D:\Users\hp\Desktop\实验数据\20250218实验yolo10 epoch745\验证结果\exp2\predictions.json'

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=DEFAULT_ANNO_JSON, help='training model path')
    parser.add_argument('--pred_json', type=str, default=DEFAULT_PRED_JSON, help='data yaml path')
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    # 打印路径信息，方便调试
    print(f"Annotation JSON path: {anno_json}")
    print(f"Prediction JSON path: {pred_json}")

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='result')