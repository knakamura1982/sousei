[sample.py]

・実行コマンド例
  python sample.py -g=0 -e=50 -b=100 -v=5

・オプション
  -g: GPU ID．指定しなかった場合 -1 となり，CPUモードで動作する
  -e: 総エポック数（学習データを各1回ずつ用いてパラメータを更新する過程を1エポックとして，それを何回繰り返すか）．
      指定しなかった場合，デフォルト値として 50 が設定される．
  -b: ミニバッチあたりの学習データ数．指定しなかった場合，デフォルト値として 100 が設定される．
  -v: 何エポックに1回の割合で識別境界を可視化するか．
      指定しなかった場合，デフォルト値として 5 が設定される．


[mnist_train.py]

・実行コマンド例
  python mnist_train.py -g=0 -e=10 -b=100 -m=mnist_model.pth

・オプション
  -g: sample.py と同じ．
  -e: sample.py と同じ（ただしデフォルト値は 10 となっている）．
  -b: sample.py と同じ．
  -m: ここで指定したパスのファイルに学習後のモデルが上書き保存される（存在しない場合は自動作成される）．
      指定しなかった場合，デフォルト値として mnist_model.pth が設定される．

・備考： 各エポック終了後のモデルが mnist_models/ 以下に自動保存される．


[mnist_predict.py]

・実行コマンド例
  python mnist_predict.py -g=0 -i=dataset/MNIST/test_data/00000.png -m=mnist_model.pth
  python mnist_predict.py -g=0 -b=100 -m=mnist_model.pth

・オプション
  -g: sample.py と同じ．
  -i: 入力画像のファイルパス．
      指定しなかった場合，ソースコード中に記載された評価用データセットに対する認識精度を求めるモードになる．
  -b: -iオプションが指定されなかった場合のみ有効．機能は sample.py と同じ．
  -m: モデルファイルのパス．ここで指定されたモデルファイルをロードして処理を実行する．
      必須項目であり，指定しなかった場合はエラー終了する．


[compress_train.py]

・実行コマンド例
  python compress_train.py -g=0 -e=10 -b=100 -f=32 -m=compress_model.pth

・オプション
  -g: sample.py と同じ．
  -e: sample.py と同じ（ただしデフォルト値は 10 となっている）．
  -b: sample.py と同じ．
  -f: 画像圧縮後のベクトルの次元数（圧縮表現に対応する中間層のユニット数）．
      指定しなかった場合，デフォルト値として 32 が設定される．
  -m: mnist_train.py と同じ（ただしデフォルト値は compress_model.pth となっている）．

・備考： 各エポック終了後のモデルおよび圧縮・復元結果の例が compress_models/ 以下に自動保存される．


[compress_exec.py]

・実行コマンド例
  python compress_exec.py -c -g=0 -f=32 -i=dataset/MNIST/test_data/00000.png -o=compress_result.csv -m=comporess_model.pth
  python compress_exec.py -d -g=0 -f=32 -i=compress_result.csv -o=decompress_result.png -m=compress_model.pth

・オプション
  -c: 指定すると圧縮処理が実行される
  -d: 指定すると復元処理が実行される．
      -cオプションと同時に指定された場合は-dオプションが優先され，復元処理が実行される．
      -cと-dの何れもが指定されなかった場合は圧縮処理が実行される．
  -g: sample.py と同じ．
  -f: compress_train.py と同じ．
      ただし，compress_train.py の実行時と異なる値が指定された場合はエラーになる．
  -i: 入力ファイルパス（圧縮時は画像ファイル，復元時は.csvファイル）
      必須項目であり，指定しなかった場合はエラー終了する．
  -o: 出力ファイルパス（圧縮時は.csvファイル，復元時は画像ファイル）
      必須項目であり，指定しなかった場合はエラー終了する．
  -m: mnist_predict.py と同じ．


[colorize_train.py]

・実行コマンド例
  python colorize_train.py -g=0 -e=10 -b=100 -m=colorize_model.pth

・オプション
  -g: sample.py と同じ．
  -e: sample.py と同じ（ただしデフォルト値は 10 となっている）．
  -b: sample.py と同じ．
  -m: mnist_train.py と同じ（ただしデフォルト値は colorize_model.pth となっている）．

・備考： 各エポック終了後のモデルおよびカラー化結果の例が colorize_models/ 以下に自動保存される．


[colorize_exec.py]

・実行コマンド例
  python colorize_exec.py -g=0 -i=dataset/VGGFace2/test_data/00000.png -o=colorize_result.png -m=colorize_model.pth 

・オプション
  -g: sample.py と同じ．
  -i: 入力画像のファイルパス．カラー画像が指定されても自動的にグレースケール化して読み込まれる．
      必須項目であり，指定しなかった場合はエラー終了する．
  -o: 出力画像（カラー化結果）のファイルパス．
      必須項目であり，指定しなかった場合はエラー終了する．
  -m: mnist_predict.py と同じ．


[upsampling_train.py]

・実行コマンド例
  python upsampling_train.py -g=0 -e=10 -b=100 -m=upsampling_model.pth

・オプション
  -g: sample.py と同じ．
  -e: sample.py と同じ（ただしデフォルト値は 10 となっている）．
  -b: sample.py と同じ．
  -m: mnist_train.py と同じ（ただしデフォルト値は upsampling_model.pth となっている）．

・備考： 各エポック終了後のモデルおよびアップサンプリング結果の例が upsampling_models/ 以下に自動保存される．


[upsampling_exec.py]

・実行コマンド例
  python upsampling_exec.py -g=0 -i=dataset/VGGFace2/test_data/00000.png -o=upsampling_result.png -m=upsampling_model.pth

・オプション
  -g: sample.py と同じ．
  -i: 入力画像のファイルパス．指定された画像の縦幅・横幅が半分になるように自動的にリサイズされる．
      必須項目であり，指定しなかった場合はエラー終了する．
  -o: 出力画像（アップサンプリング結果）のファイルパス．
      必須項目であり，指定しなかった場合はエラー終了する．
  -m: mnist_predict.py と同じ．


[cnn.py]

・備考： mnist_train.py および mnist_predict.py で使用するDNNのネットワーク構造が定義されている．


[autoencoders.py]

・備考： compress_train.py 〜 upsampling_exec.py で使用するオートエンコーダのネットワーク構造が定義されている．


