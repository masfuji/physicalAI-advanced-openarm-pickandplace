# physicalAI-advanced-openarm-pickandplace

OpenArm を MuJoCo と Meta Quest 3 で遠隔操作して Cube の pick and place データを集め、ACT で学習して評価するプロジェクトです。Gemini Robotics-ER 2 で Cube の位置を確認し、ACT の苦手な奥の範囲にあれば手前へ押してから ACT を実行します。

提出動画: https://youtu.be/qCmFcOuSQBM

## 構成

| パス | 内容 |
|---|---|
| `external/dora-openarm-vr/` | Enactic の本家（submodule）。編集しません |
| `src/openarm_pick_place/` | 追加した dora ノード（カメラ表示、ランダム配置ボタン、収集 UI） |
| `config/` | データフローとシーン。収集データは `config/data/` に保存されます |
| `scripts/` | 評価、Gemini 判定、可視化、データ変換、学習の補助スクリプト |
| `training/` | 学習と評価の環境（別の仮想環境） |

データセット、モデル、ログ、`.env` は git に含めません。

## セットアップ（Windows）

必要なもの:

- [Git](https://git-scm.com/) と [uv](https://docs.astral.sh/uv/getting-started/installation/)。Python は uv が用意します
- 学習と評価には NVIDIA GPU（CUDA 12.8 対応のドライバー）。学習は VRAM 16 GB で確認しました
- データ収集には Meta Quest 3

PowerShell で実行します。

```powershell
git clone --recurse-submodules https://github.com/masfuji/physicalAI-advanced-openarm-pickandplace.git
cd physicalAI-advanced-openarm-pickandplace
uv sync
Copy-Item .env.example .env   # Gemini を使う場合はキーを記入
```

シーンは `.venv/share/openarm_mujoco` を参照するので、仮想環境はリポジトリ直下に作ります。submodule なしで clone した場合は `git submodule update --init` を実行してください。

データセットと学習済みモデルは公開していません。評価するには、データを収集して学習してください。

## データ収集

Quest アプリの接続先に、その PC の IP アドレスを設定します。UDP 5006 番を Windows ファイアウォールで許可してください。そのうえでリポジトリ直下で実行します。

```powershell
uv run dora build config/dataflow-mujoco-pick-place-collection.yaml --uv
uv run dora run config/dataflow-mujoco-pick-place-collection.yaml --uv
```

ブラウザで http://127.0.0.1:8000 を開くと、記録の開始、終了、Cube のランダム配置を操作できます。データは `config/data/pick_place_cube_right/` に保存されます。

## 学習と評価

[training/README.md](training/README.md) を参照してください。

## ライセンス

このリポジトリは MIT です。ただし `external/dora-openarm-vr` と、Enactic のファイルを元にした `config/` のファイル（各ファイル冒頭に表示）は Apache-2.0 です。
