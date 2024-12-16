# 模型训练和验证

# 准备模型和数据目录
os.makedirs("models", exist_ok=True)
os.chdir("models")
wgethttps://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar -xf ch_PP-OCRv3_det_distill_train.tar
os.chdir("..")

# 训练检测模型
python tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams
    Global.save_model_dir=output/CCPD/det/
    Global.eval_batch_step="[0, 90]"
    Optimizer.lr.name=Const
    Optimizer.lr.learning_rate=0.0005
    Optimizer.lr.warmup_epoch=0
    Train.dataset.data_dir="CCPD2020/PPOCR"
    Train.dataset.label_file_list=["PPOCR/train/rec.txt"]
    Eval.dataset.data_dir="CCPD2020/PPOCR"
    Eval.dataset.label_file_list=["PPOCR/test/rec.txt"]

# 验证检测模型
python tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Eval.dataset.data_dir="CCPD2020/PPOCR" \
    Eval.dataset.label_file_list=["PPOCR/test/rec.txt"]

# 导出检测模型
python tools/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml -o \
    Global.pretrained_model=output/CCPD/det/best_accuracy.pdparams \
    Global.save_inference_dir="output/CCPD/det/infer"

# 测试检测模型
python3 tools/infer/predict_det.py --det_algorithm="DB" --det_model_dir="output/CCPD/det/infer" --image_dir="src.jpg" --use_gpu=True

# 训练识别模型
python tools/train.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=models/ch_PP-OCRv3_rec_train/student.pdparams \
    Global.save_model_dir="output/CCPD/rec/" \
    Global.eval_batch_step="[0, 90]" \
    Optimizer.lr.name=Const \
    Optimizer.lr.learning_rate=0.0005 \
    Optimizer.lr.warmup_epoch=0 \
    Train.dataset.data_dir="CCPD2020/PPOCR" \
    Train.dataset.label_file_list=["PPOCR/train/rec.txt"] \
    Eval.dataset.data_dir="CCPD2020/PPOCR" \
    Eval.dataset.label_file_list=["PPOCR/test/rec.txt"]

# 验证识别模型
python tools/eval.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Eval.dataset.data_dir="CCPD2020/PPOCR" \
    Eval.dataset.label_file_list=["PPOCR/test/rec.txt"]

# 导出识别模型
python tools/export_model.py -c configs/rec/PP-OCRv3/ch_PP-OCRv3_rec.yml -o \
    Global.pretrained_model=output/CCPD/rec/best_accuracy.pdparams \
    Global.save_inference_dir="output/CCPD/rec/infer"

# 测试识别模型
python3 tools/infer/predict_rec.py --image_dir="PPOCR/test/crop_imgs" \
    --rec_model_dir="output/CCPD/rec/infer" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/ppocr_keys_v1.txt"

# 模型联合推理
python tools/infer/predict_system.py \
    --det_model_dir="output/CCPD/det/infer/" \
    --rec_model_dir="output/CCPD/rec/infer/" \
    --image_dir="src.jpg" \
    --rec_image_shape=3,48,320
