## .kiro/specs/model-architecture/design.md
```markdown
# Model Architecture Design
---
priority: 1
---

## Architecture Overview

```mermaid
graph TD
    A[Input: (B, 60, 7)] --> B[Linear Embedding]
    B --> C[+ Positional Encoding]
    C --> D[LayerNorm]
    
    D --> E[Transformer Block 1]
    E --> F[Transformer Block 2]
    F --> G[...]
    G --> H[Transformer Block 6]
    
    H --> I[Attention Pooling]
    
    I --> J[Price Head]
    I --> K[Volatility Head]
    I --> L[Quantile Heads]
    
    J --> M[Price Predictions (B, 5)]
    K --> N[Volatility (B, 5)]
    L --> O[Confidence Intervals (B, 5, 5)]
```

## Detailed Component Design

### Transformer Block Architecture
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, return_attention=False):
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if return_attention:
            return x, attn_weights
        return x
```

### Positional Encoding Design
```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=60, d_model=256):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        return x + self.pos_embedding[:, :seq_len, :]
```

### Multi-Task Output Heads
```python
class PredictionHeads(nn.Module):
    def __init__(self, d_model=256, forecast_horizon=5):
        super().__init__()
        
        # Price prediction head
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, forecast_horizon),
            nn.Softplus()  # Ensure positive volatility
        )
        
        # Quantile regression heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(d_model, forecast_horizon)
            for _ in [0.1, 0.25, 0.5, 0.75, 0.9]
        ])
```

### Attention Mechanism Details
```python
class InterpretableAttention(nn.Module):
    """Custom attention with interpretability features"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and split heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (B, H, L, D)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention_weights
```

## Optimization Strategies

### Gradient Checkpointing
```python
class CheckpointedTransformer(nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.training and i > 0:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
```

### Mixed Precision Training
```python
def train_with_amp(model, data_loader, optimizer):
    scaler = GradScaler()
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
```

### Model Quantization
```python
def quantize_model(model):
    """Quantize model for deployment"""
    model.eval()
    
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.MultiheadAttention},
        dtype=torch.qint8
    )
    
    return quantized_model
```

## Inference Optimization

### TorchScript Compilation
```python
def compile_for_inference(model):
    """Compile model with TorchScript"""
    model.eval()
    
    example_input = torch.randn(1, 60, 7)
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    return traced_model
```

### Batched Inference
```python
class BatchedInferenceEngine:
    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size
        self.cache = {}
        
    @torch.no_grad()
    def predict_batch(self, inputs):
        # Check cache
        cached_results = []
        uncached_inputs = []
        uncached_indices = []
        
        for i, inp in enumerate(inputs):
            cache_key = self._compute_hash(inp)
            if cache_key in self.cache:
                cached_results.append((i, self.cache[cache_key]))
            else:
                uncached_inputs.append(inp)
                uncached_indices.append(i)
        
        # Process uncached
        if uncached_inputs:
            batch = torch.stack(uncached_inputs)
            predictions = self.model(batch)
            
            # Update cache
            for inp, pred in zip(uncached_inputs, predictions):
                cache_key = self._compute_hash(inp)
                self.cache[cache_key] = pred
        
        # Combine results
        return self._combine_results(cached_results, predictions, uncached_indices)
```
```

