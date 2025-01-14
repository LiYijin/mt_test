# mt_test

支持摩尔线程的8个模型和量化模型的测试

```shell
# 1. 创建docker 容器
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash

# 2. 准备模型
所有模型放入/models中

/models/
├── ECAPA
│   └── voxceleb_ECAPA512.onnx
├── Retinaface
│   └── RetinaFace.onnx
├── arcface
│   └── arcfaceresnet100-8.onnx
├── fastspeech2
│   ├── fastspeech2_csmsc_am_decoder.onnx
│   ├── fastspeech2_csmsc_am_encoder_infer.onnx
│   ├── fastspeech2_csmsc_am_postnet.onnx
│   └── mb_melgan_csmsc.onnx
├── hrnet
│   └── hrnet_w18_fp32.onnx
├── slowfast
│   └── slowfast.onnx
└── yolov8
    └── yolov8n.onnx

# 3. 安装onnxruntime-musa
pip install onnxruntime-1.18.1-cp38-cp38-linux_x86_64.whl 

# 4. 运行每个模型对应的测试脚本
python test_arcface.py /models/arcface/arcfaceresnet100-8.onnx
python test_ecapa.py /models/ECAPA/voxceleb_ECAPA512.onnx
python test_hrnet.py /models/hrnet/hrnet_w18_fp32.onnx
python test_retinaface.py /models/Retinaface/RetinaFace.onnx
python test_slowfast.py /models/slowfast/slowfast.onnx
python test_yolov8.py /models/yolov8/yolov8n.onnx
python test_fastspeech2_encoder.py /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
python test_fastspeech2_decoder.py /models/fastspeech2/fastspeech2_csmsc_am_decoder.onnx
python test_fastspeech2_postnet.py /models/fastspeech2/fastspeech2_csmsc_am_postnet.onnx
python test_mb_melgan.py /models/fastspeech2/mb_melgan_csmsc.onnx
```shell