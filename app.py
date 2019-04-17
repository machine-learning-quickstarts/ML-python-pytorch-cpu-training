#!/usr/bin/python
import os
import subprocess
import torch
import torchvision

dummy_input = torch.randn(10, 3, 224, 224, device='cpu')   # or cuda
model = torchvision.models.alexnet(pretrained=True).cpu()  # or cuda()

input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)

# Now synchronise the model into the service codebase
try:
    targetsvc = os.path.expandvars('$TARGET_SERVICE_NAME')
    targetrepo = os.path.expandvars('$TARGET_SERVICE_REPO')
    paths = {'subpath': targetsvc, 'repo': targetrepo}
    synccommands = '''
    git config --global credential.helper store
    jx step git credentials
    rm -rf {subpath}
    git clone {repo}
    cd {subpath}
    git checkout syncmodel || git checkout -b syncmodel
    git lfs install
    git lfs track "*.onnx"
    cp ../model.onnx .
    git add model.onnx
    git commit -m \"feat: New model trained\"
    git push --set-upstream origin syncmodel
    hub pull-request --no-edit
    '''.format(**paths)
    subprocess.run([synccommands], shell=True, check=True)

except subprocess.CalledProcessError as err:
    print('Synchronising model failed ', err)
