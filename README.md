
# PromptGroundBoxBench

PromptGroundBoxBench is a reproducible mini benchmark for prompt driven zero shot bounding box detection on a single class COCO style dataset.
It compares

* Grounding DINO via Hugging Face Transformers
* Ultralytics YOLO11x as a closed set baseline

The current focus is vehicles only.
We convert the original COCO style dataset to a single category with id 1 and name vehicle, then run COCOeval plus extra analysis metrics.

## What is included

* COCO style evaluation using pycocotools COCOeval for bbox
* Unified output format written as COCO detection JSON for both engines
* A lightweight analysis pass that produces additional statistics
  * TP FP FN counts at a fixed IoU threshold
  * Precision Recall F1 across a score threshold grid
  * Matched IoU summary for true positives
  * Score summary for matched predictions
  * Per image count summaries for GT preds TP FP FN

## Repository layout

Key files

* tools/make_coco_single_class.py
  * Converts a COCO instances json into a single class dataset with category id 1
* tools/eval_vehicles_A.py
  * Runs inference for Grounding DINO and YOLO
  * Writes predictions json for each engine
  * Runs COCOeval and writes report json
* tools/analyze_vehicles_report.py
  * Reads the predictions json and the ground truth annotations
  * Computes TP FP FN and threshold sweep statistics at a chosen IoU threshold

Default run folder in this repo

* runs/vehicles_A

## Setup

### 1 Create the conda environment

From the repository root

```bash
conda env create -f environment.yml
conda activate PromptGroundBox
````

### 2 Install a GPU compatible PyTorch build

PyTorch is intentionally not pinned inside the environment file.
Install a build that matches your CUDA runtime and GPU.

Example for a recent CUDA wheel index

```bash
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

## Dataset

### Expected dataset layout

This benchmark expects a COCO style root like this

```
data/vehicles_coco_vehicle/
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json
  train2017/
    *.jpg
  val2017/
    *.jpg
  test2017/
    *.jpg
```

Notes

* Category id must be 1
* Category name must be vehicle
* All annotations must use category_id 1

### Create the single class annotations

If you already have the original COCO style vehicles dataset under

```
data/vehicles_coco/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
  annotations/instances_test2017.json
```

Create the single class annotation files

```powershell
python tools\make_coco_single_class.py --src data\vehicles_coco\annotations\instances_train2017.json --dst data\vehicles_coco_vehicle\annotations\instances_train2017.json
python tools\make_coco_single_class.py --src data\vehicles_coco\annotations\instances_val2017.json   --dst data\vehicles_coco_vehicle\annotations\instances_val2017.json
python tools\make_coco_single_class.py --src data\vehicles_coco\annotations\instances_test2017.json  --dst data\vehicles_coco_vehicle\annotations\instances_test2017.json
```

Make sure your image folders exist under data/vehicles_coco_vehicle/train2017 val2017 test2017.
If your images are still under data/vehicles_coco you can copy or symlink them into the vehicle root.

Quick sanity check

```powershell
python -c "import json; p=r'data\vehicles_coco_vehicle\annotations\instances_val2017.json'; d=json.load(open(p,'r',encoding='utf-8')); print('cats',d['categories']); print('unique cat_ids in anns',sorted({a['category_id'] for a in d['annotations']}))"
```

Expected

* categories contains exactly one entry with id 1 and name vehicle
* unique category ids in annotations is [1]

## Running evaluation

### Model choices

Grounding DINO model id is passed by CLI.
For the large model used in our current runs use

* openmmlab-community/mm_grounding_dino_large_all

YOLO weights default to

* yolo11x.pt

### Clean start

If you want to remove all previous outputs and rerun from scratch

```powershell
Remove-Item -Recurse -Force runs\vehicles_A
New-Item -ItemType Directory -Force runs\vehicles_A | Out-Null
```

### Run COCOeval for all splits using Grounding DINO large

```powershell
cd D:\programming\PromptGroundBox

python tools\eval_vehicles_A.py --root data\vehicles_coco_vehicle --split train2017 --prompt "vehicle" --device cuda --gd_model_id "openmmlab-community/mm_grounding_dino_large_all" --out_dir runs/vehicles_A
python tools\eval_vehicles_A.py --root data\vehicles_coco_vehicle --split val2017   --prompt "vehicle" --device cuda --gd_model_id "openmmlab-community/mm_grounding_dino_large_all" --out_dir runs/vehicles_A
python tools\eval_vehicles_A.py --root data\vehicles_coco_vehicle --split test2017  --prompt "vehicle" --device cuda --gd_model_id "openmmlab-community/mm_grounding_dino_large_all" --out_dir runs/vehicles_A
```

Outputs written per split

* runs/vehicles_A/preds_gd_<split>.json
* runs/vehicles_A/preds_yolo_<split>.json
* runs/vehicles_A/report_<split>.json

Each report includes

* COCOeval metrics for both engines
* mean and p95 latency for both engines

## Running extra analysis metrics

This step uses the predictions json from eval_vehicles_A.py.
It computes fixed IoU matching statistics and threshold sweeps.

Run with IoU threshold 0.50

```powershell
python tools\analyze_vehicles_report.py --root data\vehicles_coco_vehicle --split train2017 --run_dir runs/vehicles_A --iou_thr 0.50
python tools\analyze_vehicles_report.py --root data\vehicles_coco_vehicle --split val2017   --run_dir runs/vehicles_A --iou_thr 0.50
python tools\analyze_vehicles_report.py --root data\vehicles_coco_vehicle --split test2017  --run_dir runs/vehicles_A --iou_thr 0.50
```

Outputs written per split

* runs/vehicles_A/analysis_<split>_iou50.json

## Current raw results

These are the raw numbers from our current run configuration.
Grounding DINO is the large model.

### COCOeval metrics

#### Train2017

| Model                  | AP     | AP50   | AP75   | AP Small | AP Medium | AP Large | AR@100 | Mean Latency (s/img) |
| ---------------------- | ------ | ------ | ------ | -------- | --------- | -------- | ------ | -------------------- |
| Grounding DINO (Large) | 0.3348 | 0.3744 | 0.3506 | 0.0291   | 0.0938    | 0.7452   | 0.3740 | 0.3624               |
| YOLO11x                | 0.1854 | 0.2549 | 0.1970 | 0.0522   | 0.1328    | 0.3505   | 0.3120 | 0.0286               |

---

#### Val2017

| Model                  | AP     | AP50   | AP75   | AP Small | AP Medium | AP Large | AR@100 |
| ---------------------- | ------ | ------ | ------ | -------- | --------- | -------- | ------ |
| Grounding DINO (Large) | 0.1880 | 0.1966 | 0.1970 | 0.0000   | 0.0000    | 0.4303   | 0.1959 |
| YOLO11x                | 0.2212 | 0.2696 | 0.2463 | 0.1183   | 0.1475    | 0.3639   | 0.3311 |

---

#### Test2017

| Model                  | AP     | AP50   | AP75   | AP Small | AP Medium | AP Large | AR@100 |
| ---------------------- | ------ | ------ | ------ | -------- | --------- | -------- | ------ |
| Grounding DINO (Large) | 0.2157 | 0.2209 | 0.2218 | 0.0000   | 0.0000    | 0.4749   | 0.2155 |
| YOLO11x                | 0.2333 | 0.2932 | 0.2567 | 0.1185   | 0.1686    | 0.3702   | 0.3363 |

---

## Extra Analysis at IoU = 0.50

**Best F1 point on score threshold grid**

---

#### Train2017

| Model          | Best Score Thr | Precision | Recall | F1    | TP  | FP  | FN  | IoU Mean | IoU P50 | IoU P90 |
| -------------- | -------------- | --------- | ------ | ----- | --- | --- | --- | -------- | ------- | ------- |
| Grounding DINO | 0.0            | 0.785     | 0.410  | 0.539 | 325 | 89  | 468 | 0.935    | 0.970   | 0.989   |
| YOLO11x        | 0.3            | 0.407     | 0.381  | 0.393 | 302 | 440 | 491 | 0.863    | 0.906   | 0.976   |

---

#### Val2017

| Model          | Best Score Thr | Precision | Recall | F1    | TP | FP  | FN  |
| -------------- | -------------- | --------- | ------ | ----- | -- | --- | --- |
| Grounding DINO | 0.0            | 0.762     | 0.427  | 0.547 | 93 | 29  | 125 |
| YOLO11x        | 0.3            | 0.423     | 0.390  | 0.406 | 85 | 116 | 133 |

---

#### Test2017

| Model          | Best Score Thr | Precision | Recall | F1    | TP | FP | FN |
| -------------- | -------------- | --------- | ------ | ----- | -- | -- | -- |
| Grounding DINO | 0.0            | 0.852     | 0.411  | 0.554 | 46 | 8  | 66 |
| YOLO11x        | 0.0            | 0.379     | 0.420  | 0.398 | 47 | 77 | 65 |

---

## How to interpret the current behavior

* Grounding DINO large tends to produce fewer boxes per image and much higher precision.
  When it detects a vehicle, localization quality is very high, matched IoU is typically above 0.93 on train.
  The main limitation is recall, many GT boxes are missed.
* YOLO11x produces more boxes per image and therefore many false positives on this dataset setup.
  It is much faster, but the precision recall tradeoff suggests the baseline weights and thresholds are not well aligned with this data distribution.

## Visual reports
## Visualize

This repo writes all plots and qualitative overlays under the run directory so the README can directly embed images from the repository.

For the current run the folder is

`runs/vehicles_A/visuals/<split>/`

Where `<split>` is one of `train2017`, `val2017`, `test2017`.

Each split folder includes

* `index.html` a one page visual dashboard (open locally in a browser)
* `table_cocoeval_metrics.png` COCOeval metrics table
* `table_best_f1_iou50.png` best F1 point at IoU 0.50
* PR curves and F1 versus threshold plots
* latency summary plot
* histograms for score and IoU distributions and per image TP FP FN
* `qualitative/` overlays, one pair per image id, with ground truth in green and predictions in red

### Quick open

Open the HTML dashboard

* `runs/vehicles_A/visuals/train2017/index.html`
* `runs/vehicles_A/visuals/val2017/index.html`
* `runs/vehicles_A/visuals/test2017/index.html`

### Summary plots

Train2017

![COCOeval metrics table](runs/vehicles_A/visuals/train2017/table_cocoeval_metrics.png)

![Best F1 table at IoU 0.50](runs/vehicles_A/visuals/train2017/table_best_f1_iou50.png)

![AP family](runs/vehicles_A/visuals/train2017/bars_ap_family.png)

![AP by area](runs/vehicles_A/visuals/train2017/bars_ap_by_area.png)

![AR100](runs/vehicles_A/visuals/train2017/bars_ar100.png)

![Latency](runs/vehicles_A/visuals/train2017/latency_bars.png)

![PR Grounding DINO](runs/vehicles_A/visuals/train2017/pr_grounding_dino.png)

![F1 vs threshold Grounding DINO](runs/vehicles_A/visuals/train2017/pr_grounding_dino_f1.png)

![PR YOLO](runs/vehicles_A/visuals/train2017/pr_yolo.png)

![F1 vs threshold YOLO](runs/vehicles_A/visuals/train2017/pr_yolo_f1.png)

Val2017

![COCOeval metrics table](runs/vehicles_A/visuals/val2017/table_cocoeval_metrics.png)

![Best F1 table at IoU 0.50](runs/vehicles_A/visuals/val2017/table_best_f1_iou50.png)

![AP family](runs/vehicles_A/visuals/val2017/bars_ap_family.png)

![AP by area](runs/vehicles_A/visuals/val2017/bars_ap_by_area.png)

![AR100](runs/vehicles_A/visuals/val2017/bars_ar100.png)

![Latency](runs/vehicles_A/visuals/val2017/latency_bars.png)

![PR Grounding DINO](runs/vehicles_A/visuals/val2017/pr_grounding_dino.png)

![F1 vs threshold Grounding DINO](runs/vehicles_A/visuals/val2017/pr_grounding_dino_f1.png)

![PR YOLO](runs/vehicles_A/visuals/val2017/pr_yolo.png)

![F1 vs threshold YOLO](runs/vehicles_A/visuals/val2017/pr_yolo_f1.png)

Test2017

![COCOeval metrics table](runs/vehicles_A/visuals/test2017/table_cocoeval_metrics.png)

![Best F1 table at IoU 0.50](runs/vehicles_A/visuals/test2017/table_best_f1_iou50.png)

![AP family](runs/vehicles_A/visuals/test2017/bars_ap_family.png)

![AP by area](runs/vehicles_A/visuals/test2017/bars_ap_by_area.png)

![AR100](runs/vehicles_A/visuals/test2017/bars_ar100.png)

![Latency](runs/vehicles_A/visuals/test2017/latency_bars.png)

![PR Grounding DINO](runs/vehicles_A/visuals/test2017/pr_grounding_dino.png)

![F1 vs threshold Grounding DINO](runs/vehicles_A/visuals/test2017/pr_grounding_dino_f1.png)

![PR YOLO](runs/vehicles_A/visuals/test2017/pr_yolo.png)

![F1 vs threshold YOLO](runs/vehicles_A/visuals/test2017/pr_yolo_f1.png)

### Error distribution plots

These help explain whether performance issues are mostly missing objects (FN) or hallucinated boxes (FP)

Train2017

![Per image FN Grounding DINO](runs/vehicles_A/visuals/train2017/hist_per_image_fn_grounding_dino.png)
![Per image FP Grounding DINO](runs/vehicles_A/visuals/train2017/hist_per_image_fp_grounding_dino.png)

![Per image FN YOLO](runs/vehicles_A/visuals/train2017/hist_per_image_fn_yolo.png)
![Per image FP YOLO](runs/vehicles_A/visuals/train2017/hist_per_image_fp_yolo.png)

Val2017

![Per image FN Grounding DINO](runs/vehicles_A/visuals/val2017/hist_per_image_fn_grounding_dino.png)
![Per image FP Grounding DINO](runs/vehicles_A/visuals/val2017/hist_per_image_fp_grounding_dino.png)

![Per image FN YOLO](runs/vehicles_A/visuals/val2017/hist_per_image_fn_yolo.png)
![Per image FP YOLO](runs/vehicles_A/visuals/val2017/hist_per_image_fp_yolo.png)

Test2017

![Per image FN Grounding DINO](runs/vehicles_A/visuals/test2017/hist_per_image_fn_grounding_dino.png)
![Per image FP Grounding DINO](runs/vehicles_A/visuals/test2017/hist_per_image_fp_grounding_dino.png)

![Per image FN YOLO](runs/vehicles_A/visuals/test2017/hist_per_image_fn_yolo.png)
![Per image FP YOLO](runs/vehicles_A/visuals/test2017/hist_per_image_fp_yolo.png)

### Qualitative overlays

Overlays are saved as pairs in `runs/vehicles_A/visuals/<split>/qualitative/`

* `*_gd.png` Grounding DINO overlay
* `*_yolo.png` YOLO overlay

Ground truth boxes are green and predicted boxes are red.

Below are a few side by side examples (HTML table, works on GitHub)

Train2017

<table>
  <tr>
    <th>Grounding DINO</th>
    <th>YOLO11x</th>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000003_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000003_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000076_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000076_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000220_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/train2017/qualitative/000220_yolo.png" width="420"></td>
  </tr>
</table>

Val2017

<table>
  <tr>
    <th>Grounding DINO</th>
    <th>YOLO11x</th>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000006_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000006_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000024_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000024_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000059_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/val2017/qualitative/000059_yolo.png" width="420"></td>
  </tr>
</table>

Test2017

<table>
  <tr>
    <th>Grounding DINO</th>
    <th>YOLO11x</th>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000004_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000004_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000016_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000016_yolo.png" width="420"></td>
  </tr>
  <tr>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000029_gd.png" width="420"></td>
    <td><img src="runs/vehicles_A/visuals/test2017/qualitative/000029_yolo.png" width="420"></td>
  </tr>
</table>




## License

MIT, see LICENSE.

