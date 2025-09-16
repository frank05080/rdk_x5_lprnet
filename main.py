import cv2
import numpy as np
import bpu_infer_lib

img_path = 'rec_img/1.jpg'
img_data = cv2.imread(img_path)
img_data = img_data[:,:,::-1]  # BGR to RGB

img_data = cv2.resize(img_data,(94, 24))
img_data = (img_data - 127.5) / 127.5 
img_data = np.transpose(img_data, (2,0,1))  # HWC to CHW
img_data = np.expand_dims(img_data, 0)  # to BCHW
np_data = np.array(img_data, dtype=np.float32)

inf = bpu_infer_lib.Infer(False)
inf.load_model("lpr.bin")
inf.read_input(np.ascontiguousarray(np_data, dtype=np.float32), 0)
inf.forward(more=True)
inf.get_output()
res = inf.outputs[0].data
res = res.reshape(1,68,18)

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

def reprocess(pred):
    pred_data = pred[0]
    pred_label = np.argmax(pred_data, axis=0)
    no_repeat_blank_label = []
    pre_c = pred_label[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in pred_label:  # dropout repeate label and blank label
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    char_list = [CHARS[i] for i in no_repeat_blank_label]
    return ''.join(char_list)

plate_str = reprocess(res)
print(plate_str)