# Flower Hierarchical FL Development

## When to Use
Flower (flwr) を使った連合学習コードの追加・修正時。

## Key Patterns

### 入れ子サーバー構造
Edge Node の `fit()` 内で `start_server()` を呼ぶ。パラメータの capture には `evaluate_fn` callback を使う。

### 集約戦略の追加
1. `src/strategies/aggregation.py` に新クラスを作成
2. `@register_strategy("name")` デコレータで登録
3. `config/global.yaml` の `strategy.name` で切り替え

### Flower API の注意
- `start_server` / `start_client` は deprecated だが入れ子構造では必須
- 新しい `ServerApp` / `ClientApp` API は使わない
- `NumPyClient` を継承し、`to_client()` で変換して `start_client` に渡す

### パラメータ変換
```python
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
# NDArrays → Parameters (Proto): ndarrays_to_parameters(arrays)
# Parameters (Proto) → NDArrays: parameters_to_ndarrays(params)
```
