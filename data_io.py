import os
import csv
import cv2
import numpy as np


# CSV形式で記載されたデータファイルを読み込む
# 左端の列にクラスラベルが，残りの列に特徴ベクトルが記載されていることを想定
#   - filename: データファイルのファイルパス
#   - with_header: ヘッダ行が存在するか否か
#   - dic: クラスラベル名からクラス番号への対応を表す辞書（Noneの場合は自動作成）
def read_features(filename, with_header=True, dic=None):

    # CSVファイルを開く
    f = open(filename, 'r')
    reader = csv.reader(f)

    # ヘッダ行が存在する場合は空読みする
    if with_header:
        next(reader)

    # データファイルを読み込む
    lab = []
    feat = []
    name = []
    if dic == None:
        # クラスラベル名とクラス番号の対応関係が未知の場合
        nc = 0
        dic = {}
        for row in reader:
            if not row[0] in dic:
                name.append(row[0])
                dic[row[0]] = nc
                nc += 1
            lab.append(dic[row[0]])
            feat.append(np.asarray(row[1:], dtype=np.float32))
    else:
        # クラスラベル名とクラス番号の対応関係が既知の場合
        for row in reader:
            if not row[0] in dic:
                continue
            lab.append(dic[row[0]])
            feat.append(np.asarray(row[1:], dtype=np.float32))

    # 読み込み結果を numpy.ndarray 形式に変換
    lab = np.asarray(lab)
    feat = np.asarray(feat)

    # CSVファイルを閉じる
    f.close()

    # 結果を返す
    #   - lab: ラベル配列
    #   - feat: 特徴量配列
    #   - dic: クラスラベル名からクラス番号への対応を表す辞書
    #   - name: クラスラベル名配列（dic != Noneのときは空）
    return lab, feat, dic, name


# CSV形式で記載された画像ファイルリストを読み込む
#   - filename: 画像リストファイルのファイルパス
#   - data_dir: 画像ファイル名の先頭に付加する接頭辞（画像フォルダ名）
#   - with_header: ヘッダ行が存在するか否か
#   - cid_lab: 何列目にクラスラベルが記載されているかを指定する整数値（0-based index, デフォルト値: 0）
#   - cid_dat: 何列目に画像ファイル名が記載されているかを指定する整数値（0-based index, デフォルト値: 1）
#   - dic: クラスラベル名からクラス番号への対応を表す辞書（Noneの場合は自動作成）
def read_image_list(filename, data_dir='./', with_header=True, cid_lab=0, cid_dat=1, dic=None):

    # CSVファイルを開く
    f = open(filename, 'r')
    reader = csv.reader(f)

    # ヘッダ行が存在する場合は空読みする
    if with_header:
        next(reader)

    # データファイルを読み込む
    lab = []
    img = []
    name = []
    if dic == None:
        # クラスラベル名とクラス番号の対応関係が未知の場合
        nc = 0
        dic = {}
        for row in reader:
            if not row[cid_lab] in dic:
                name.append(row[cid_lab])
                dic[row[cid_lab]] = nc
                nc += 1
            lab.append(dic[row[cid_lab]])
            img.append(os.path.join(data_dir, row[cid_dat]))
    else:
        # クラスラベル名とクラス番号の対応関係が既知の場合
        for row in reader:
            if not row[cid_lab] in dic:
                continue
            lab.append(dic[row[cid_lab]])
            img.append(os.path.join(data_dir, row[cid_dat]))

    # 読み込み結果を numpy.ndarray 形式に変換
    lab = np.asarray(lab)

    # CSVファイルを閉じる
    f.close()

    # 結果を返す
    #   - lab: ラベル配列
    #   - img: 画像ファイル名のリスト
    #   - dic: クラスラベル名からクラス番号への対応を表す辞書
    #   - name: クラスラベル名配列（dic != Noneのときは空）
    return lab, img, dic, name


# 画像 img にごま塩ノイズを追加する
#   - img: 画像（numpy ndarray）
#   - mode: 読み込みモード（0: グレースケール画像モード，それ以外: カラー画像モード）
#   - num: ごま塩ノイズとして入れる点の個数
def add_noise(img, mode, num):
    w_num = num // 2
    b_num = num - w_num
    pts_x = np.random.randint(0, img.shape[1] - 1, w_num)
    pts_y = np.random.randint(0, img.shape[0] - 1, w_num)
    if mode == 0:
        img[(pts_y, pts_x)] = 255
    else:
        img[(pts_y, pts_x)] = (255, 255, 255)
    pts_x = np.random.randint(0, img.shape[1] - 1, b_num)
    pts_y = np.random.randint(0, img.shape[0] - 1, b_num)
    if mode == 0:
        img[(pts_y, pts_x)] = 0
    else:
        img[(pts_y, pts_x)] = (0, 0, 0)
    return img


# 単一の画像を読み込む
#   - filename: 画像ファイルのファイルパス
#   - size: 画像サイズを強制的に size = (width, height) に変更する（size == None のときは無視される）
#   - pad: 画像サイズの強制変更をパディングで実現するか否か（pad == True ならパディングで，pad == False なら単純な伸縮でサイズを変更する．size == None のときは無視される）
#   - mode: 読み込みモード（0: グレースケール画像モード，それ以外: カラー画像モード）
#   - with_noise: True なら強制的にごま塩ノイズを追加して読み込む
#   - n_noise_points: ごま塩ノイズとして追加する点の個数（with_noise == False のときは無視される）
def load_single_image(filename, size=None, pad=False, mode=1, with_noise=False, n_noise_points=100):

    if mode == 0:
        # グレースケール画像モード
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # グレースケール画像として読み込む
        if not size is None:
            if pad:
                dw = img.shape[1] - size[0]
                dh = img.shape[0] - size[1]
                dwh = dw // 2
                dhh = dh // 2
                if dw < 0:
                    temp = img[0, 0] * np.ones((img.shape[0], size[0]), dtype=img.dtype)
                    temp[ : , -dwh : -dwh + img.shape[1]] = img
                    dw = dwh = 0
                    img = temp
                if dh < 0:
                    temp = img[0, 0] * np.ones((size[1], img.shape[1]), dtype=img.dtype)
                    temp[-dhh : -dhh + img.shape[0], : ] = img
                    dh = dhh = 0
                    img = temp
                img = img[dhh : dhh + size[1], dwh : dwh + size[0]]
            else:
                img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_NEAREST)
        if with_noise:
            img = add_noise(img, mode, n_noise_points)
        img = img.reshape((1, img.shape[0], img.shape[1])) # チャンネル数 1 の三次元テンソルとなるように変形
        img = np.asarray(img, dtype=np.float32) / 255 # データ形式を 32bit float に変更し，画素値が [0,1] の範囲に収まるように正規化
    else:
        # カラー画像モード
        img = cv2.imread(filename, cv2.IMREAD_COLOR) # カラー画像として読み込む
        if not size is None:
            if pad:
                dw = img.shape[1] - size[0]
                dh = img.shape[0] - size[1]
                dwh = dw // 2
                dhh = dh // 2
                if dw < 0:
                    temp = np.asarray([[img[0, 0]]]) * np.ones((img.shape[0], size[0], img.shape[2]), dtype=img.dtype)
                    temp[ : , -dwh : -dwh + img.shape[1], : ] = img
                    dw = dwh = 0
                    img = temp
                if dh < 0:
                    temp = np.asarray([[img[0, 0]]]) * np.ones((size[1], img.shape[1], img.shape[2]), dtype=img.dtype)
                    temp[-dhh : -dhh + img.shape[0], : , : ] = img
                    dh = dhh = 0
                    img = temp
                img = img[dhh : dhh + size[1], dwh : dwh + size[0]]
            else:
                img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_NEAREST)
        if with_noise:
            img = add_noise(img, mode, n_noise_points)
        img = img.transpose(2, 0, 1) # (チャンネル数，縦幅，横幅) の順となるように変形
        img = np.asarray(img, dtype=np.float32) / 255 # データ形式を 32bit float に変更し，画素値が [0,1] の範囲に収まるように正規化

    return img


# 画像データを読み込む
#   - image_list: 画像ファイル名のリスト（関数 read_image_list の出力を指定する）
#   - ids: リスト中の何番目の画像を読み込むかを指定する整数配列
#   - size: 画像サイズを強制的に size = (width, height) に変更する（size == None のときは無視される）
#   - pad: 画像サイズの強制変更をパディングで実現するか否か（pad == True ならパディングで，pad == False なら単純な伸縮でサイズを変更する．size == None のときは無視される）
#   - mode: 読み込みモード（0: グレースケール画像モード，それ以外: カラー画像モード）
#   - with_noise: True なら強制的にごま塩ノイズを追加して読み込む
#   - n_noise_points: ごま塩ノイズとして追加する点の個数（with_noise == False のときは無視される）
def load_images(image_list, ids=None, size=None, pad=False, mode=1, with_noise=False, n_noise_points=100):
   
    # 画像をロード
    n = 0
    images = [] # 読み込んだ画像の集合
    if ids is None:
        for i in range(0, len(image_list)):
            img = load_single_image(image_list[i], size, pad, mode, with_noise, n_noise_points) # i番目の画像を読み込む
            images.append(img) # 画像を配列に追加
            n += 1 # 読み込んだ画像の総数をカウントしておく
    else:
        for i in range(0, len(ids)):
            img = load_single_image(image_list[ids[i]], size, pad, mode, with_noise, n_noise_points) # ids[i]番目の画像を読み込む
            images.append(img) # 画像を配列に追加
            n += 1 # 読み込んだ画像の総数をカウントしておく

    # 読み込み結果を numpy.ndarray に変換して出力
    images = np.asarray(images, dtype=np.float32)

    # 結果を返却
    return images


# ラベルデータを one-hot vector 形式に展開する
#   - label_list: ラベルデータのリスト（関数 read_image_list の出力を指定する）
#   - ids: リスト中の何番目のデータを展開するかを指定する整数配列
#   - n_classes: ラベルの種類数の総数（0以下の場合は自動計算）
def load_labels(label_list, ids=None, n_classes=0):

    if n_classes <= 0:
        n_classes = len(np.unique(label_list))

    n = 0
    labels = []
    if ids is None:
        for i in range(0, len(label_list)):
            lab = np.zeros(n_classes, dtype=np.float32)
            lab[label_list[i]] = 1.0
            labels.append(lab)
            n += 1
    else:
        for i in range(0, len(ids)):
            lab = np.zeros(n_classes, dtype=np.float32)
            lab[label_list[ids[i]]] = 1.0
            labels.append(lab)
            n += 1

    labels = np.asarray(labels, dtype=np.float32)

    return labels
