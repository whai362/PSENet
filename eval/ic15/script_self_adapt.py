import os
import mmcv
import argparse

parser = argparse.ArgumentParser(description='Hyperparams')
# parser.add_argument('--gt', nargs='?', type=str, default=None)
parser.add_argument('--pred', nargs='?', type=str, default=None)
args = parser.parse_args()

output_root = '../outputs/tmp_results/'
pred = mmcv.load(args.pred)

def write_result_as_txt(image_name, bboxes, path, words=None):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = path + 'res_%s.txt'%(image_name)
    lines = []
    for i, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        if words is None:
            line = "%d,%d,%d,%d,%d,%d,%d,%d\n"%tuple(values)
            lines.append(line)
        elif words[i] is not None:
            line = "%d,%d,%d,%d,%d,%d,%d,%d"%tuple(values) + ",%s\n"%words[i]
            lines.append(line)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)

def eval(thr):
    for key in pred:
        pred_ = pred[key]
        line_num = len(pred_['scores'])
        bboxes = []
        # words = []
        for i in range(line_num):
            if pred_['scores'][i] < thr:
                continue
            bboxes.append(pred_['bboxes'][i])
            # words.append(pred_['words'][i])

        write_result_as_txt(key, bboxes, output_root)

    cmd = 'cd %s;zip -j %s %s/*' % ('../outputs/', 'tmp_results.zip', 'tmp_results')
    res_cmd = os.popen(cmd)
    res_cmd.read()

    cmd = 'cd ic15 && python2 script.py -g=gt.zip -s=../../outputs/tmp_results.zip && cd ..'
    res_cmd = os.popen(cmd)
    res_cmd = res_cmd.read()
    h_mean = float(res_cmd.split(',')[-2].split(':')[-1])
    return res_cmd, h_mean

max_h_mean = 0
best_thr = 0
best_res = ''
for i in range(85, 100):
    thr = float(i) / 100
    # print('Testing thr: %f'%thr)
    res, h_mean = eval(thr)
    # print(thr, h_mean)
    if h_mean > max_h_mean:
        max_h_mean = h_mean
        best_thr = thr
        best_res = res

print('thr: %f | %s'%(best_thr, best_res))
