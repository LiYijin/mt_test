## ort-for-musa phase 2 test

### Steps

```shell
# run docker container
docker run -it --privileged --shm-size=80G  --pid=host --network=host --env MTHREADS_VISIBLE_DEVICES=all --shm-size=80g ort-musa-cmcc:v0.1-cmcc /bin/bash

# install onnxruntime-musa
pip install onnxruntime-1.18.1-cp38-cp38-linux_x86_64.whl 

# test models (test scripts are in the repo), taks arcface as example
python test_arcface.py /model/path/to/model/onnx
```
