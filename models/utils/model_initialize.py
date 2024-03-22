import onnx

model = onnx.load('../arcface_w600k_r50.onnx')

for initializer in model.graph.initializer:
    print(initializer.name, initializer.dims)