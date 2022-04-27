from speechCommandModel import *

# Load pretrained model
model = CNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Convert to ONNX format
x = torch.empty(1, 1, 98, 50)
torch.onnx.export(model,
                      x,
                      "cmdRecognitionPyTorch.onnx",
                      export_params=True,
                      opset_version=9,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])