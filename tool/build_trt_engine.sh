# #!/bin/bash
# /usr/src/tensorrt/bin/trtexec --onnx=./model/pointpillar.onnx --fp16 --plugins=build/libpointpillar_core.so --saveEngine=./model/pointpillar.plan --inputIOFormats=fp16:chw,int32:chw,int32:chw --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed > model/pointpillar.8611.log 2>&1
#!/bin/bash
# /usr/src/tensorrt/bin/trtexec \
#   --onnx=./model/pointpillar.onnx \
#   --fp16 \
#   --staticPlugins=build/libpointpillar_core.so \
#   --saveEngine=./model/pointpillar.plan \
#   --inputIOFormats=fp16:chw,int32:chw,int32:chw \
#   --verbose --dumpLayerInfo --dumpProfile --separateProfileRun --profilingVerbosity=detailed \
#   > model/pointpillar.trt10.log 2>&1
# 1. 경로 설정 (사용자님 환경에 맞춤)
ONNX_PATH="/home/a/CUDA-PointPillars/model/pointpillar.onnx"
ENGINE_PATH="/home/a/CUDA-PointPillars/model/pointpillar.plan"
PLUGIN_PATH="/home/a/CUDA-PointPillars/build/libpointpillar_core.so"

# 2. TensorRT 변환 실행
# 로그를 파일로 숨기지 않고 화면에 바로 출력하도록 수정했습니다.
/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --fp16 \
  --staticPlugins=$PLUGIN_PATH \
  --saveEngine=$ENGINE_PATH \
  --inputIOFormats=fp16:chw,int32:chw,int32:chw \
  --verbose \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --profilingVerbosity=detailed