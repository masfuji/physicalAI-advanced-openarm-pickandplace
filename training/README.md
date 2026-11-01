# 学習と評価

コマンドはすべてこのフォルダで PowerShell から実行します。先にリポジトリ直下で `uv sync` を済ませてください（シーンが参照するファイルが入ります）。

```powershell
uv sync
```

`torch` などを取得するので、初回は数 GB のダウンロードがあります。

データセットと学習済みモデルは公開していません。以下のコマンドのパスは、収集したデータから自分で作る場合のものです。

## データ変換

Windows では、公式の変換コマンドの代わりに `../scripts/convert_dataset_windows.py` を使います。引数は `openarm-dataset-convert` と同じです。動画の変換には `ffmpeg` が PATH に必要です。入っていなければ、仮想環境に同梱のものを使えます。

```powershell
Copy-Item (.venv\Scripts\python.exe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") .venv\Scripts\ffmpeg.exe
```

GPU メモリが足りないときは、`../scripts/downscale_lerobot_dataset.py` で動画の解像度を下げます。

## 学習

Windows ではシンボリックリンクの権限エラーを避けるため、`../scripts/train_act_windows.py` を使います。

```powershell
.venv/Scripts/python.exe ../scripts/train_act_windows.py `
--dataset.repo_id=local/pick_place_cube --dataset.root=data/pick_place_cube_lerobot_half `
--dataset.video_backend=pyav --policy.type=act --policy.push_to_hub=false `
--policy.device=cuda --policy.use_amp=true `
--policy.optimizer_lr=1e-5 --policy.optimizer_lr_backbone=1e-5 `
--batch_size=8 --num_workers=8 --steps=100000 --log_freq=200 --save_freq=10000 `
--wandb.enable=false --output_dir=outputs/act_pick_place_v1
```

## 評価

```powershell
.venv/Scripts/python.exe ../scripts/evaluate_act.py `
outputs/act_pick_place_v1/checkpoints/100000/pretrained_model `
--cube-y=0 --start home --episodes 20 --output outputs/eval/home_start
```

| オプション | 内容 |
|---|---|
| `--cube-y=Y` | Cube の出現中心。省略時はシーンの値 |
| `--start home` / `--start dataset --dataset DIR` | 腕の開始姿勢 |
| `--temporal-ensemble-coeff=0.05` | temporal ensembling を使う |
| `--viewer` | MuJoCo のウィンドウで実時間で見る |
| `--gemini-judge` | 各試行の終了時に Gemini Robotics-ER 2 で成否を判定して記録する（`.env` にキーが必要） |
| `--max-retries N` | 失敗したら最大 N 回再実行する。手を横へよけて Cube を押し、ホームへ戻ってから ACT を再度動かす |
| `--retry-with sim` / `gemini` | 失敗の判定と Cube の位置確認に、シミュレータの状態か Gemini を使う |
| `--slide-target X Y` | Cube を押す目標位置。既定は学習時の配置に合わせた 0.53 -0.025 |
| `--no-slide` | Cube を押さずに再実行する |
| `--no-sim-stop` | シミュレータの状態で試行を打ち切らず、毎回 `--horizon-seconds` まで動かす |

## 提出版の評価

Gemini で Cube の位置を確認し、x 0.555 以上なら目標（x 0.515、y -0.01）へ押してから ACT を実行します。再実行はしません。`.env` にキーが必要です。

```powershell
.venv/Scripts/python.exe ../scripts/evaluate_planned.py `
outputs/act_pick_place_v1/checkpoints/100000/pretrained_model `
--episodes 100 --seed 3100 --output outputs/eval_planned/plan_far
```

比較用の ACT 単体は次のコマンドです。

```powershell
.venv/Scripts/python.exe ../scripts/evaluate_act.py `
outputs/act_pick_place_v1/checkpoints/100000/pretrained_model `
--cube-y=0 --start home --episodes 100 --seed 3100 --horizon-seconds 45 --output outputs/eval_planned/act_only_45s
```

## ACT を使わない版

Gemini が認識、判断、計画、判定を行い、スクリプトの動作関数を呼び出します。

```powershell
.venv/Scripts/python.exe ../scripts/gemini_orchestrate.py --seed 3000 --output outputs/orchestrate/try
```

## 分析

`evaluate_act.py` の結果の場所は、各スクリプトの冒頭に書いてあります。

| スクリプト | 内容 |
|---|---|
| `../scripts/choose_slide_target.py` | ACT 単体の結果から、押す目標の候補を比べる |
| `../scripts/plot_success_map.py` | Cube の位置ごとの成功率マップを描く |
| `../scripts/analyze_dataset_positions.py` | 学習データの Cube の初期位置を映像から推定する |
| `../scripts/plot_dataset_trajectories.py` | 学習データの軌道の 3D 図を描く（[dataset_trajectories.html](dataset_trajectories.html)） |

## 可視化と判定

```powershell
# チャンク別の軌道
uv run --no-sync python ../scripts/plot_eval_trajectories.py outputs/eval/home_start --output outputs/eval/chunks.html

# 収集データの軌道
uv run --no-sync python ../scripts/plot_trajectories.py ../config/data/pick_place_cube/dataset

# 保存済みの評価結果を Gemini で判定
.venv/Scripts/python.exe ../scripts/gemini_judge.py outputs/eval/home_start
```
