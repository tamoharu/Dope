from pathlib import Path

import onnx
import onnxoptimizer

onnxfile = Path("../inswapper_128.onnx")
onnx_model = onnx.load(onnxfile)
passes = ["eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)
output_onnx_path = f"{onnxfile.parent}/{onnxfile.stem}_clean.onnx"
onnx.save(optimized_model, output_onnx_path)