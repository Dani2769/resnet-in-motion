"""Train and export the playground's real residual digit classifier.

Reproduce: python scripts/train-classifier.py
No external downloads: sklearn's bundled UCI optical handwritten digits.
The test split is evaluated only after validation chooses an epoch.
"""
from pathlib import Path
import copy
import json
import time
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EPOCHS = 90
torch.set_num_threads(2)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x, shortcut_scale=1., branch_scale=1., record=False):
        a = F.relu(self.conv1(x))
        branch = self.conv2(a)
        shortcut = x * shortcut_scale
        scaled_branch = branch * branch_scale
        summed = shortcut + scaled_branch
        output = F.relu(summed)
        if record:
            return output, dict(input=x, conv1=a, branch=branch,
                                scaledBranch=scaled_branch, shortcut=shortcut,
                                sum=summed, output=output)
        return output


class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(3)])
        self.head = nn.Linear(512, 10)

    def forward(self, x, shortcut_scales=(1., 1., 1.), branch_scales=(1., 1., 1.), record=False):
        inp = x
        x = F.relu(self.stem(x))
        stem = x
        records = []
        for i, block in enumerate(self.blocks):
            if record:
                x, values = block(x, shortcut_scales[i], branch_scales[i], True)
                records.append(values)
            else:
                x = block(x, shortcut_scales[i], branch_scales[i])
        flat = x.flatten(1)
        logits = self.head(flat)
        if record:
            return dict(input=inp, stem=stem, blocks=records, flat=flat,
                        logits=logits, probabilities=F.softmax(logits, dim=1),
                        prediction=int(logits.argmax(1).item()))
        return logits


def encode(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, dict):
        return {key: encode(val) for key, val in value.items()}
    if isinstance(value, list):
        return [encode(val) for val in value]
    return value


def main():
    started = time.monotonic()
    digits = load_digits()
    indices = np.arange(len(digits.target))
    development, test = train_test_split(indices, test_size=.2, stratify=digits.target, random_state=SEED)
    train, validation = train_test_split(development, test_size=.2, stratify=digits.target[development], random_state=SEED)
    x = torch.tensor(digits.images[:, None] / 16., dtype=torch.float32)
    y = torch.tensor(digits.target, dtype=torch.long)
    net = ResidualClassifier()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=.00005)
    rng = np.random.default_rng(SEED)
    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = rng.permutation(train)
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            logits = net(x[batch])
            loss = F.cross_entropy(logits, y[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        with torch.no_grad():
            val_logits = net(x[validation])
            val_loss = float(F.cross_entropy(val_logits, y[validation]))
            val_accuracy = float((val_logits.argmax(1) == y[validation]).float().mean())
        history.append(dict(epoch=epoch, validationLoss=val_loss, validationAccuracy=val_accuracy))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(net.state_dict())
        if epoch % 10 == 0:
            print(json.dumps(dict(epoch=epoch, validationLoss=round(val_loss, 5), validationAccuracy=round(val_accuracy, 5))), flush=True)
    net.load_state_dict(best_state)
    net.eval()
    # The shipped inference and parity fixtures use these same rounded weights.
    weights = {key: np.round(value.numpy().astype('float64'), 8).reshape(-1).tolist()
               for key, value in best_state.items()}
    net = net.double()
    with torch.no_grad():
        for key, parameter in net.named_parameters():
            parameter.copy_(torch.tensor(weights[key], dtype=torch.float64).reshape(parameter.shape))
    xd = x.double()
    with torch.no_grad():
        train_logits = net(xd[train])
        val_logits = net(xd[validation])
        test_logits = net(xd[test])
        test_probs = F.softmax(test_logits, dim=1)
        test_predictions = test_logits.argmax(1).numpy()
        train_accuracy = float((train_logits.argmax(1) == y[train]).double().mean())
        validation_accuracy = float((val_logits.argmax(1) == y[validation]).double().mean())
        test_accuracy = float((test_logits.argmax(1) == y[test]).double().mean())
        test_loss = float(F.cross_entropy(test_logits, y[test]))
    samples = [dict(id=int(index), label=int(digits.target[index]),
                    pixels=(digits.images[index] / 16.).reshape(-1).tolist(),
                    expectedPrediction=int(test_predictions[j]),
                    expectedProbabilities=np.round(test_probs[j].numpy(), 10).tolist())
               for j, index in enumerate(test)]
    meta = dict(name='Mini ResNet · handwritten digits', seed=SEED,
                dataset='UCI Optical Recognition of Handwritten Digits / sklearn load_digits',
                datasetURL='https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
                datasetSize=len(indices), trainCount=len(train), validationCount=len(validation), testCount=len(test),
                inputSize=8, channels=8, blockCount=3, parameterCount=sum(p.numel() for p in net.parameters()),
                architecture='Conv 1→8 (3×3, padding 1) → ReLU → 3 × [Conv 8→8 (3×3, padding 1) → ReLU → Conv 8→8 (3×3, padding 1) + identity → ReLU] → Flatten 512 → Linear 10 → Softmax',
                residualActivation='ReLU after addition', normalization='none',
                convolution='cross-correlation, stride 1, zero padding 1',
                optimizer='AdamW', learningRate=.002, weightDecay=.0001, batchSize=64,
                epochs=EPOCHS, selectedEpoch=best_epoch,
                selection='Lowest validation cross-entropy; the held-out test set was evaluated after selection.',
                sampleSelection='All 360 held-out samples, in deterministic split order. The gallery uses the first samples of each label, without selecting on predictions.',
                split='Stratified 80/20 development/test; development stratified 80/20 train/validation, both random_state=42.',
                limitations='Small teaching architecture, not ResNet-50. Softmax probabilities are uncalibrated. Shortcut/branch changes are inference-time interventions without retraining, not a fair architecture benchmark. Hand drawings may differ from the training distribution.',
                weightDecimals=8, torchVersion=torch.__version__, sklearnVersion=sklearn.__version__)
    metrics = dict(trainAccuracy=train_accuracy, validationAccuracy=validation_accuracy,
                   testAccuracy=test_accuracy, testCorrect=int((test_predictions == digits.target[test]).sum()),
                   testLoss=test_loss, confusionMatrix=confusion_matrix(digits.target[test], test_predictions).tolist())
    data = dict(meta=meta, weights=weights, samples=samples, metrics=metrics)
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False)
    (ROOT / 'src' / 'model' / 'model-data.js').write_text(
        '/* Trained parameters and held-out UCI digits. Reproduce: python scripts/train-classifier.py */\n'
        '(function(root){"use strict";const data=' + encoded +
        ';if(typeof module!=="undefined"&&module.exports)module.exports=data;root.ResNetData=data;'
        '})(typeof window!=="undefined"?window:globalThis);\n', encoding='utf-8', newline='\n')
    fixtures = []
    # Full layer-by-layer traces also verify signed branch values and ReLU order.
    cases = [(0, [1, 1, 1], [1, 1, 1]), (1, [0, 1, 1], [1, 1, 1]),
             (2, [1, 1, 1], [0, 0, 0]), (3, [.4, 1, .8], [.7, 1, 1.3])]
    with torch.no_grad():
        for sample_index, shortcut_scales, branch_scales in cases:
            record = encode(net(xd[test[sample_index]:test[sample_index] + 1],
                                shortcut_scales, branch_scales, True))
            fixtures.append(dict(sampleIndex=sample_index, shortcutScales=shortcut_scales,
                                 branchScales=branch_scales, expected=record))
    (ROOT / 'tests' / 'classifier-fixtures.json').write_text(json.dumps(fixtures, separators=(',', ':')), encoding='utf-8')
    summary = dict(meta=meta, metrics=metrics, trainingHistory=history,
                   trainingSeconds=round(time.monotonic() - started, 2))
    (ROOT / 'reports' / 'classifier-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, meta=meta, metrics=metrics, seconds=summary['trainingSeconds'])), flush=True)


if __name__ == '__main__':
    main()
