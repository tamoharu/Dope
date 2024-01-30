import onnxruntime

model = onnxruntime.InferenceSession('./enhance_face.py/codeformer.onnx')

input = model.get_inputs()
output = model.get_outputs()

print('input')
for i in input:
    print(f'{i.name}: {i.shape}')

print('\noutput')
for o in output:
    print(f'{o.name}: {o.shape}')