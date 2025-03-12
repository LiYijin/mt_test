XPU_TYPE="cuda"     # Should be 'cuda' or 'musa'

echo "==arcface=="
python test_arcface.py --xpu $XPU_TYPE --model /models/arcface/arcfaceresnet100-8.onnx
echo "==ECAPA=="
python test_ecapa.py --xpu $XPU_TYPE --model /models/ECAPA/voxceleb_ECAPA512.onnx
echo "==hrnet=="
python test_hrnet.py --xpu $XPU_TYPE --model /models/hrnet/hrnet_w18_fp32.onnx
echo "==Retinaface=="
python test_retinaface.py --xpu $XPU_TYPE --model /models/Retinaface/RetinaFace.onnx
echo "==slowfast=="
python test_slowfast.py --xpu $XPU_TYPE --model /models/slowfast/slowfast.onnx
echo "==yolov8=="
python test_yolov8.py --xpu $XPU_TYPE --model /models/yolov8/yolov8n.onnx
echo "==fastspeech2_decoder=="
python test_fastspeech2_decoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_decoder.onnx
echo "==fastspeech2_encoder=="
python test_fastspeech2_encoder.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_encoder_infer.onnx
echo "==fastspeech2_postnet=="
python test_fastspeech2_postnet.py --xpu $XPU_TYPE --model /models/fastspeech2/fastspeech2_csmsc_am_postnet.onnx
echo "==mb_melgan=="
python test_mb_melgan.py --xpu $XPU_TYPE --model /models/fastspeech2/mb_melgan_csmsc.onnx

# MUSA Test Only
echo "==resnet_quant=="
python test_resnet_q.py --model /models/resnet-q/resnet-q.onnx --dataset /datasets/resnet-q