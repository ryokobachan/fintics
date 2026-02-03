# Test PyPI 公開手順

このドキュメントでは、finticsをTest PyPIに公開し、テストする手順を説明します。

## 📋 準備完了チェックリスト

以下の準備が完了しています：
- ✅ バージョン: 1.0.0（全ファイルで統一済み）
- ✅ 作者メール: fintics.org@gmail.com
- ✅ requirements.txt: 本番用依存関係のみ
- ✅ requirements-dev.txt: 開発用依存関係
- ✅ MANIFEST.in: 作成済み
- ✅ pyproject.toml: 完全なメタデータ
- ✅ setup.py: 改善済み
- ✅ ビルド成果物: dist/fintics-1.0.0-py3-none-any.whl および fintics-1.0.0.tar.gz
- ✅ パッケージチェック: PASSED

## 🔐 Step 1: Test PyPI アカウント作成

### 1.1 アカウント登録

1. https://test.pypi.org/account/register/ にアクセス
2. 以下の情報を入力：
   - Username: お好きなユーザー名
   - Email: fintics.org@gmail.com（または任意のメールアドレス）
   - Password: 強力なパスワード
3. メール認証を完了

### 1.2 APIトークンの作成

**セキュリティのため、パスワードではなくAPIトークンを使用します**

1. https://test.pypi.org/manage/account/ にログイン
2. "API tokens" セクションまでスクロール
3. "Add API token" をクリック
4. Token name: `fintics-upload`（任意の名前）
5. Scope: "Entire account (all projects)" を選択
6. "Add token" をクリック
7. **🔴 重要**: トークンをコピーして安全な場所に保存
   - トークンは `pypi-` で始まる長い文字列です
   - この画面を閉じると二度と表示されません！

## 📤 Step 2: Test PyPI にアップロード

### オプション A: 対話的アップロード（推奨・初回）

```bash
cd /Users/kobayashiryotaro/Developer/Fintics_Project/Fintics
python -m twine upload --repository testpypi dist/*
```

プロンプトが表示されます：
```
Enter your username: __token__
Enter your password: <ここにAPIトークンを貼り付け>
```

**注意**: 
- Username は必ず `__token__` と入力（アンダースコア2つ + token + アンダースコア2つ）
- Password には Step 1.2 で取得したAPIトークンを貼り付け

### オプション B: 環境変数を使用（2回目以降）

```bash
# APIトークンを環境変数に設定（このセッションのみ有効）
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_API_TOKEN_HERE

# アップロード
python -m twine upload --repository testpypi dist/*
```

### 成功メッセージの例

```
Uploading distributions to https://test.pypi.org/legacy/
Uploading fintics-1.0.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.5/74.5 kB • 0:00:01
Uploading fintics-1.0.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.3/59.3 kB • 0:00:01

View at:
https://test.pypi.org/project/fintics/1.0.0/
```

## 🧪 Step 3: Test PyPI からテストインストール

### 3.1 仮想環境を作成

```bash
# プロジェクト外の場所で実行（例: ホームディレクトリ）
cd ~
mkdir fintics-test
cd fintics-test
python -m venv test-env
source test-env/bin/activate  # macOS/Linux
```

### 3.2 Test PyPI からインストール

**重要**: TA-Lib は事前にシステムレベルでインストールが必要です

```bash
# TA-Lib のインストール（macOS）
brew install ta-lib

# Test PyPI からfinticsをインストール
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fintics
```

**注意**: 
- `--index-url`: Test PyPIから取得
- `--extra-index-url`: 依存パッケージは通常のPyPIから取得

### 3.3 インストール確認

```bash
# バージョン確認
python -c "import fintics; print(fintics.__version__)"
# 出力: 1.0.0

# CLIコマンド確認
fintics --help

# 簡単なテスト
fintics strategy list
```

### 3.4 後片付け

```bash
deactivate
cd ~
rm -rf fintics-test
```

## ✅ Step 4: 動作確認が完了したら

Test PyPIでの動作確認が完了したら、本番PyPIへの公開に進めます。

### 本番 PyPI へのアップロード準備

1. **本番PyPIアカウント作成**: https://pypi.org/account/register/
2. **APIトークン作成**: https://pypi.org/manage/account/
3. **パッケージ名の確認**: https://pypi.org/project/fintics/ にアクセスして、名前が使用可能か確認

### 本番 PyPI へアップロード

```bash
cd /Users/kobayashiryotaro/Developer/Fintics_Project/Fintics

# 本番PyPIにアップロード
python -m twine upload dist/*
```

プロンプトが表示されたら：
```
Enter your username: __token__
Enter your password: <本番PyPIのAPIトークン>
```

### 本番PyPIからインストール

```bash
pip install fintics
```

## 📝 トラブルシューティング

### エラー: "The user '...' isn't allowed to upload to project 'fintics'"

→ パッケージ名がすでに使用されています。pyproject.tomlとsetup.pyの`name`を変更してください。

### エラー: "File already exists"

→ 同じバージョンは再アップロードできません。バージョン番号を上げてください：
```bash
# バージョンを上げる（例: 1.0.0 → 1.0.1）
# fintics/__init__.py の __version__ を変更
# ビルドし直す
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine upload --repository testpypi dist/*
```

### エラー: TA-Lib のインストールに失敗

TA-Libは特別なインストールが必要です：

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**Windows:**
- https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib から whl ファイルをダウンロード
- `pip install TA_Lib‑0.4.xx‑cp3xx‑cp3xx‑win_amd64.whl`

READMEにインストール手順を追加することを推奨します。

## 🎉 完了！

Test PyPIでの公開とテストが完了しました！

次のステップ:
1. ✅ Test PyPIでの動作確認
2. 📝 必要に応じてドキュメント改善
3. 🚀 本番PyPIへ公開
4. 📢 GitHubでリリースタグを作成
5. 🌟 README更新・コミュニティへの共有
