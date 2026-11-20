# ResNet · follow an image

One continuous 3D playground answers one question: **what happens to a handwritten digit as it travels through a residual network?** Choose a real test image, follow its representations through three residual blocks, see both paths reunite at each addition, and arrive at the classifier’s prediction. Clicking a block opens its computation directly in the 3D scene.

The small network is a teaching architecture, not ResNet-50. Predictions and feature values come from exported trained weights. Travelling lights explain the order of computation; they are not measured hardware activity.

## Open the playground

Build the website and offline edition from the editable `src/` directory:

```powershell
cd resnet-shortcuts
python scripts/build-playground.py
python -m http.server 8000 --directory dist --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. The application uses static HTML, CSS, and JavaScript, with no package installation or runtime machine-learning dependency.

For the single-file offline edition:

```powershell
python scripts/build-playground.py
```

Open `ResNet_3D_Playground.html` directly. The builder embeds the stylesheet and five scripts in the HTML loading order: model data, inference, lesson configuration, the renderer, and the controller. The hosted page’s Google Fonts import is removed in this edition; system fonts are used without network requests. The builder does not retrain the network.

## One scene, one story

Choose a digit from the held-out gallery and use the primary **Next** button, which names its destination, to set the pace. The explanation, **Back**, **Next**, restart, and **Play/Pause/Replay walkthrough** controls sit above the scene and remain sticky on desktop. They control a 19-checkpoint journey:

1. Normalize the 8 × 8 image into 64 values.
2. Apply the stem convolution and ReLU to produce eight feature maps.
3. Send the first block’s input into both the learned branch and identity shortcut.
4. Apply the branch’s first convolution and ReLU, then its second convolution to produce the signed learned change.
5. Add matching values from both paths to produce the raw sum.
6. Apply ReLU as a separate stage, clipping negative sums to zero.
7. Follow the same five stages through blocks two and three.
8. Flatten the final tensor into 512 values.
9. Produce ten linear scores, convert them to softmax probabilities, and predict a digit.

The checkpoints explain the computation; they are not 19 learned layers. The count is two initial stages, five stages for each of three residual blocks, and two classifier stages. **Step XX / 19** shows overall progress; **Stage N of 5** shows progress within a block. Gallery samples retain their true labels, including examples the model classifies incorrectly. **Another sample** cycles through examples of the selected digit in deterministic split order.

The route groups all 19 accessible step targets under **Input / Stem / Block 01 / Block 02 / Block 03 / Classifier**. A group label jumps to its first step. Jumping to a block section preserves the current overview or expanded presentation; the top block buttons explicitly expand a block.

Click a residual block, use a labelled block button, or press **1/2/3** to expand that block **in the 3D scene itself**. Its feature planes physically separate as the camera moves in, while unrelated parts of the network fade. A block opens at its input split rather than at the completed addition. Hovering over a collapsed block highlights its planes and shortcut as one clickable group. The expanded layout reveals input `x`, the first convolution and ReLU, the signed learned branch `F(x)`, the identity shortcut, their addition, the pre-activation sum `x + F(x)`, and the output `ReLU(x + F(x))`. Both paths visibly meet at the addition node.

Five buttons inside the viewport follow **Split → Conv 1 → Conv 2 · F(x) → Add → ReLU**. **Next** and **Back** use this same order; automatic playback follows the same checkpoints. Completed stages have a distinct state; future maps remain subdued and labelled **Later** until reached. The numeric overlay reveals the selected input, convolution response, learned change, sum, and output at their relevant stages. On phones, the same five tensors use two columns and three rows for readable maps and labels.

At **Add**, number tokens carry the selected measured `x` and `F(x)` to the addition node before the sum is revealed. At **ReLU**, the measured sum passes through the activation and produces the actual output. **Replay addition** and **Replay ReLU** repeat these short clips. **Show a clipped value** selects a real negative sum from the current block so its transition to zero can be inspected; it does not invent a value, alter the image, or modify the model.

Play/pause controls automatic stage advancement. Manual stepping and replay can run one finite explanatory animation while the automatic tour is paused, just as opening a block can run a finite camera transition. With reduced motion, values appear immediately without travelling tokens.

**Inspect one value** stays within the viewport. Choose a channel and select a pixel on an expanded feature plane, or use the row and column controls. A dark outline marks the selection, and its channel/row/column address uses indices 0–7. The numeric overlay follows that same address through input, learned branch, sum, and output. Displayed decimals are rounded; numerical additions use ≈ while the underlying tensor addition stays exact. Select **← Network** or **Escape** to return from an expanded block to the whole network. There is no further convolution drilldown or inspector panel below the scene.

Feature maps are learned representations and need not resemble the digit. Their shapes match at each addition. The overview normalizes brightness separately for each tensor. The expanded input, branch, sum, and output share a scale for the selected channel; convolution-processing maps can use their own normalization. Neither view is a probability map. Conv 1 includes ReLU; Conv 2 produces the signed change `F(x)` before the residual addition and final ReLU. Coral represents negative values, which the final ReLU clips to zero.

The page consists of a compact header and the playground. A current-view tag identifies the network or block. All ten digit probabilities remain available in the initially collapsed **All digit probabilities** disclosure. Supporting architecture, training evidence, and limitations are available in the model information dialog. There is no article, drawer, custom drawing input, or collection of experiments after the scene. The network always uses its trained shortcut and branch settings.

Mouse dragging rotates the scene; touch swipes preserve page scrolling and a tap selects a block or an exposed pixel. Ctrl/Command + mouse wheel zooms. The Network button, labelled block buttons, and zoom controls provide visible alternatives. **Reset network view** restores the overview camera and is hidden while a block is expanded; **Restart journey** returns to the first checkpoint.

Keyboard shortcuts are **Space** for play/pause, **Left/Right** for previous/next checkpoint, **R** for restart, **1/2/3** for block expansion, and **Escape** to return to the whole network. Escape also works from focused sliders. Other shortcuts defer to inputs, selects, textareas, editable content, and open dialogs. With an expanded block and the canvas focused, arrow keys move the selected pixel instead of the journey. Space keeps its native button activation when a button has focus. Tab reaches the viewport controls, channel buttons, and row/column selectors. Reduced motion makes expansion, camera changes, and operation results immediate and disables travelling pulses.

## The exact computation

```text
Input: 1 × 8 × 8, grayscale / 16
Stem: Conv(1 → 8, 3 × 3, padding 1) → ReLU

Three residual blocks:
  x: 8 × 8 × 8
  F(x): Conv(8 → 8, 3 × 3, padding 1)
        → ReLU
        → Conv(8 → 8, 3 × 3, padding 1)
  y: ReLU(x + F(x))

Flatten: 8 × 8 × 8 → 512
Classifier: Linear(512 → 10)
Display: Softmax over ten logits
```

Both paths contribute. The shortcut does not choose a route that replaces the branch. Each element of `x` is added to the corresponding element of `F(x)`. All convolutions use stride 1 and zero padding 1. There is no batch normalization, pooling, downsampling, or projection shortcut. The model has **8,714 parameters**, across seven convolutional layers and one linear layer.

The final classifier predicts the class; intermediate feature maps do not. Softmax scores are uncalibrated probabilities, not guarantees of correctness. Accuracy describes the fixed held-out split, not every possible handwritten image. This playground does not compare separately trained plain and residual architectures or claim that vanishing gradients alone explain the original degradation problem.

## Model and dataset provenance

The dataset is scikit-learn’s bundled `load_digits`: 1,797 public UCI handwritten digit images, each 8 × 8 with pixel values from 0 to 16. Normalization divides by 16. Reproducing training needs no dataset download.

| Item                                  | Value                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------- |
| Training / validation / test examples | 1,149 / 288 / 360                                                         |
| Split                                 | Stratified 80/20 development/test, then stratified 80/20 train/validation |
| Split and model seed                  | 42                                                                        |
| Optimizer                             | AdamW, learning rate 0.002, weight decay 0.0001                           |
| Schedule                              | Cosine annealing over 90 epochs to learning rate 0.00005                  |
| Batch size                            | 64                                                                        |
| Model selection                       | Lowest validation cross-entropy, epoch 15                                 |
| Training / validation accuracy        | 99.65% / 97.92%                                                           |
| Test accuracy                         | **96.39%: 347 / 360**                                                     |
| Test cross-entropy                    | 0.153269                                                                  |
| Exported parameter precision          | Eight decimal places                                                      |
| Training software                     | PyTorch 2.11.0+cu128 on CPU; scikit-learn 1.8.0                           |

The test split is evaluated after validation selects an epoch. All 360 held-out samples ship in `src/model/model-data.js`, with ground-truth labels and PyTorch reference probabilities. Examples are not selected according to correctness. `reports/classifier-summary.json` records the training history, confusion matrix, metadata, and metrics; `scripts/train-classifier.py` reproduces the model and exports.

## Source layout

Edit **`src/`**, not `dist/` or the standalone HTML. One build produces both runtime outputs.

```text
resnet-shortcuts/
  src/
    index.html                 Page structure and accessible controls
    app/
      lesson.js                Lesson text, phases, route groups, step helpers
      controller.js            State, UI events, playback, and view synchronization
    model/
      model-data.js            Generated trained weights and held-out samples
      inference.js             Pure numerical forward pass
    rendering/scene.js         Canvas renderer, camera, picking, finite animations
    styles/playground.css      Responsive presentation
  scripts/                     Build, syntax checks, source packaging, training
  tests/                       Current model/controller/renderer/build coverage
  tools/kernel-inspector.js    Development-only numerical helper
  reports/                     Reproducible training summary
  dist/                        Generated static website
  ResNet_3D_Playground.html     Generated offline edition
```

`npm run build` and `python scripts/build-playground.py` are equivalent. The build replaces only the generated `dist/` folder and refreshes the offline HTML. Runtime dependency order is declared once in `src/index.html`; the bundler reads it rather than maintaining another script list.

When you select a digit, the controller runs one forward pass and sends the resulting activations to the canvas renderer. The controller also tracks the current walkthrough step and selected cell. The renderer uses that state to display values from the same forward pass as you orbit the scene, open a block, or replay an operation.

`ResNetEngine.forward(pixels)` accepts 64 finite grayscale values and returns `input`, `stem`, `blocks`, `flat`, `logits`, `probabilities`, and `prediction`. Feature tensors are channel-first `Float64Array`s, indexed by `channel * 64 + row * 8 + col`. Each block records `input`, post-ReLU `conv1`, signed `branch`, `scaledBranch`, `shortcut`, pre-ReLU `sum`, and `output`.

The renderer’s `expandBlock(index)` and `collapseBlock()` methods change the spatial presentation without changing the model or its predictions. `setState(...)` synchronizes the selected channel and pixel, journey step, explicit `blockPhase`, motion preference, and playback state. `replayOperation()` repeats the current addition or ReLU clip. Operation-stage callbacks reveal numeric results only when the corresponding presentation stage is reached. Projected cell selection reports its row and column to the application. Expansion is an explanatory view of the same measured tensors, not another network execution or a change in architecture.

For numerical research, the inference API still accepts optional three-element `shortcutScales` and `branchScales` arrays. Tests retain coverage for these developer-level interventions. They are not controls in the playground. Block inspection uses the fixed trained network, without editing weights or path strengths.

## Reproduce and verify

Training requires Python with PyTorch, NumPy, and scikit-learn. The browser requires none of these:

```powershell
# No installation required for the app, build, or numerical tests.
python scripts/build-playground.py
node scripts/check-source.mjs
node --test tests/*.test.cjs

# Optional: reproduce model training with the Python dependencies above.
python scripts/train-classifier.py
```

For consistent development formatting, install the pinned formatter with `npm ci`. Then run:

```powershell
npm run format
npm run format:check
npm run check
npm test
npm run build
npm run package:source
```

Numerical tests compare the JavaScript engine with independent PyTorch traces from rounded exported weights, including every held-out prediction and probability. They check exact additions, ReLU placement, shape consistency, and input validation. Renderer tests exercise guided steps across viewport sizes, block picking, operation timing and measured values, replay, hover feedback, touch scrolling, camera bounds, reduced motion, and cleanup. The retained development-only kernel helper has separate tests for convolution products, padding, and trace agreement. These numerical and event tests are distinct from browser layout checks.

## References and scope

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015.
- He et al., [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027), 2016.
- UCI, [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
- scikit-learn, [`load_digits` dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
