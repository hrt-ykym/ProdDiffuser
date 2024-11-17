# ProdDiffuser

ProdDiffuserは、製品画像と背景画像を組み合わせて、プロンプトに基づいた背景生成を行うPythonアプリケーションです。

- **背景画像が用意されている場合**:
  - `assets`ディレクトリに背景画像を配置すると、製品画像と指定された背景画像が合成されます。

- **背景画像が用意されていない場合**:
  - 背景画像がない場合、[Hugging Faceのyahoo-inc/photo-background-generationモデル](https://huggingface.co/yahoo-inc/photo-background-generation)を使用して、プロンプトに基づいた背景を生成し、製品画像と合成します。

## メモ
改善PRじゃんじゃんお待ちしてます！

https://github.com/user-attachments/assets/84728dc9-00d9-4aa0-80cc-075bb65576f5

# 機能

- 製品画像と背景画像の合成
- プロンプトに基づく背景生成
- 画像のリサイズとパディング

# 必要条件
- ローカルPython環境
- Docker
- NVIDIA GPU（オプション、GPUを使用する場合）

# セットアップ

## ローカルで実行する場合
GPUを使う場合は以下を参考にCUDAをインストールしてください。
https://qiita.com/YokoPhys-h/items/274aecc84a7c42b1efb2

```bash
pip install -r requirements.txt
python src/main.py
```


## Dockerを使用する場合
GPUを使う場合
   以下を参考にしてください。
   https://qiita.com/YokoPhys-h/items/274aecc84a7c42b1efb2

1. **Dockerイメージのビルド**

   プロジェクトのルートディレクトリで以下のコマンドを実行してDockerイメージをビルドします。

   ```bash
   docker build -t prod_diffuser .
   ```

2. **Dockerコンテナの実行**

   GPUを使用する場合は以下のコマンドを実行します。

   ```bash
   docker run --rm -it --gpus all prod_diffuser
   ```

   **GPUがなくても動作します**。GPUがない場合でも、以下のコマンドで実行可能です。

   ```bash
   docker run --rm -it prod_diffuser
   ```

# ファイルについて

1. `assets`ディレクトリに製品画像と背景画像を配置します。
2. `src/main.py`内のパラメータを必要に応じて変更します。
3. 上記の手順でDockerコンテナを実行し、生成された画像が`output`ディレクトリに保存されます。

# パラメータの説明

- `product_image_path`: 製品画像のパスを指定します。
- `background_image_path`: 背景画像のパスを指定します。背景画像がない場合は、プロンプトに基づいて生成されます。
- `output_path`: 生成された画像の保存先パスを指定します。
- `prompt`: 背景生成に使用するプロンプトを指定します。背景画像がない場合に使用されます。
- `target_size`: 出力画像のサイズを指定します（幅, 高さ）。
- `scale`: 製品画像のスケールを指定します。1.0で元のサイズ、0.5で半分のサイズになります。
- `position`: 製品画像を配置する位置を指定します。`None`の場合は中央に配置されます。
- `seed`: 画像生成のランダムシードを指定します。再現性のある結果を得るために使用します。
- `num_inference_steps`: 画像生成の推論ステップ数を指定します。ステップ数が多いほど詳細な画像が生成されます。
- `controlnet_conditioning_scale`: ControlNetの条件付けスケールを指定します。生成画像の制御に影響を与えます。
