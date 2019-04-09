#!/usr/bin/python
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cpu')   # or cuda
model = torchvision.models.alexnet(pretrained=True).cpu()  # or cuda()

input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)
