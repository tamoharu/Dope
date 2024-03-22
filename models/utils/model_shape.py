from pathlib import Path

import onnxruntime

file = Path('../efficientnetv2_arcface.onnx')
model = onnxruntime.InferenceSession(file)
input = model.get_inputs()
output = model.get_outputs()
print('input')
for i in input:
    print(f'{i.name}: {i.shape}')
print('\noutput')
for o in output:
    print(f'{o.name}: {o.shape}')