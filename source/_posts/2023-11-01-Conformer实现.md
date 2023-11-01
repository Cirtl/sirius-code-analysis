---
title: Conformer实现
date: 2023-11-01 15:54:24
tags: [Paddle, Conformer]
---

# Conformer基于Paddle框架的实现

## 1. 模型结构

![](Conformer.jpg)

Conformer模型由[*Conformer: Convolution-augmented Transformer for Speech Recognition*](https://arxiv.org/abs/2005.08100)这篇论文首先提出，其架构如图所示。

基于Pytorch架构，已有多种对于Conformer的实现，本篇博客参考[Sooftware](https://github.com/sooftware/conformer)的实现，将其迁移至Paddle框架。

## 2. 模型代码

下面将分层说明Conformer代码的实现。

### 2.1 数据预处理

该部分为Conformer Block之前的部分，包含一层SepcAug、一层卷积采样（Convolution Subsampling）以及一层线性层和随机失活层。这里未实现SpecAug。

此处代码为Convolution Subsampling的代码实现：

```python
class Conv2DSubSampling(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Conv2DSubSampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs, input_lengths):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.shape

        outputs = outputs.transpose([0, 2, 1, 3])
        outputs = outputs.reshape([batch_size, subsampled_lengths, channels * sumsampled_dim])

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths
```

### 2.2 Feed Forward Module

![](FeedForward.jpg)

该模块结构如图所示，下面展示该模块代码实现。

```python
class FeedForwardModule(nn.Layer):
    def __init__(self, encoder_dim=512, expansion_factor=4, dropout_p=0.1):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(dropout_p),
            Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(dropout_p)
        )

    def forward(self, inputs):
        return self.sequential(inputs)
```

### 2.3 Multi-Headed Self Attention Module

![](SelfAttention.jpg)

该模块结构如图所示，下面展示该模块代码实现。

```python
class MultiHeadedSelfAttentionModule(nn.Layer):
    def __init__(self, d_model, num_heads, dropout_p=0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, mask=None):
        batch_size, seq_length, _ = inputs.shape
        pos_embedding = self.positional_encoding(seq_length)
        pos_embedding = paddle.fluid.layers.expand(pos_embedding, expand_times=[batch_size, 1, 1])

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)
```

该模块涉及两个自定义层，分别是`PositionalEncoding`和`RelativeMultiHeadAttention`，下面分别进行介绍。

#### 2.3.1 PositionalEncoding

该层负责对输出进行位置编码，具体实现如下：

```python
class PositionalEncoding(nn.Layer):
    def __init__(self, d_model=512, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = paddle.zeros([max_len, d_model])
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, step=2, dtype='float32') * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]
```

#### 2.3.2 RelativeMultiHeadAttention

该层负责结合位置编码输入进行相对多头注意力计算，具体实现如下：

```python
class RelativeMultiHeadAttention(nn.Layer):
    def __init__(self, d_model=512, num_heads=16, dropout_p=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = paddle.static.create_parameter(shape=[self.num_heads, self.d_head], dtype='float32', default_initializer=nn.initializer.XavierUniform())
        self.v_bias = paddle.static.create_parameter(shape=[self.num_heads, self.d_head], dtype='float32', default_initializer=nn.initializer.XavierUniform())

        self.out_proj = Linear(d_model, d_model)

    def forward(self, query, key, value, pos_embedding, mask=None):
        batch_size = value.shape[0]

        query = self.query_proj(query).reshape([batch_size, -1, self.num_heads, self.d_head])
        key = self.key_proj(key).reshape([batch_size, -1, self.num_heads, self.d_head]).transpose([0, 2, 1, 3])
        value = self.value_proj(value).reshape([batch_size, -1, self.num_heads, self.d_head]).transpose([0, 2, 1, 3])
        pos_embedding = self.pos_proj(pos_embedding).reshape([batch_size, -1, self.num_heads, self.d_head])

        content_score = paddle.matmul((query + self.u_bias).transpose([0, 2, 1, 3]), key.transpose([0, 1, 3, 2]))
        pos_score = paddle.matmul((query + self.v_bias).transpose([0, 2, 1, 3]), pos_embedding.transpose([0, 2, 3, 1]))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = paddle.matmul(attn, value).transpose([0, 2, 1, 3])
        context = context.reshape([batch_size, -1, self.d_model])

        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.shape
        zeros = paddle.zeros(shape=[batch_size, num_heads, seq_length1, 1])
        padded_pos_score = paddle.concat([zeros, pos_score], axis=-1)

        padded_pos_score = padded_pos_score.reshape([batch_size, num_heads, seq_length2 + 1, seq_length1])
        pos_score = padded_pos_score[:, :, 1:].reshape(pos_score.shape)

        return pos_score
```

### 2.4 Convolution Module

![](Convolution.jpg)

该模块结构如图所示，下面展示该模块代码实现：

```python
class ConformerConvModule(nn.Layer):
    def __init__(self, in_channels, kernel_size=31, expansion_factor=2, dropout_p=0.1):
        super(ConformerConvModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose([0, 2, 1]),
            PointwiseConv1D(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthWiseConv1D(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm1D(in_channels),
            Swish(),
            PointwiseConv1D(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(dropout_p)
        )

    def forward(self, inputs):
        return self.sequential(inputs).transpose([0, 2, 1])
```

该模块涉及两个自定义层，分别是`PointwiseConv1D`和`DepthWiseConv1D`，下面分别介绍这两个自定义层。

#### 2.4.1 PointwiseConv1D

```python
class PointwiseConv1D(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super(PointwiseConv1D, self).__init__()
        self.conv = nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias_attr=bias
        )

    def forward(self, inputs):
        return self.conv(inputs)
```

#### 2.4.2 DepthWiseConv1D

```python
class DepthWiseConv1D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthWiseConv1D, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias_attr=bias
        )

    def forward(self, inputs):
        return self.conv(inputs)
```

### 2.5 Conformer Encoder

#### 2.5.1 Conformer Block

考虑到每一个模块最后都需要进行一次残差连接，因此，我们定义了一个`ResidualConnectionModule`类，用于实现残差连接。

```python
class ResidualConnectionModule(nn.Layer):
    def __init__(self, module, module_factor=1.0, input_factor=1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs):
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)
```

在上述计算层的定义后，我们可以直接写出`ConformerBlock`的定义：

```python
class ConformerBlock(nn.Layer):
    def __init__(
        self,
        encoder_dim = 512,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1
        
        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module = FeedForwardModule(
                    encoder_dim = encoder_dim,
                    expansion_factor = feed_forward_expansion_factor,
                    dropout_p = feed_forward_dropout_p
                ),
                module_factor = self.feed_forward_residual_factor
            ),
            ResidualConnectionModule(
                module = MultiHeadedSelfAttentionModule(
                    d_model = encoder_dim,
                    num_heads = num_attention_heads,
                    dropout_p = attention_dropout_p
                )
            ),
            ResidualConnectionModule(
                module = ConformerConvModule(
                    in_channels = encoder_dim,
                    kernel_size = conv_kernel_size,
                    expansion_factor = conv_expansion_factor,
                    dropout_p = conv_dropout_p
                )
            ),
            ResidualConnectionModule(
                module = FeedForwardModule(
                    encoder_dim = encoder_dim,
                    expansion_factor = feed_forward_expansion_factor,
                    dropout_p = feed_forward_dropout_p,
                ),
                module_factor = self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs):
        return self.sequential(inputs)
```

#### 2.5.2 Conformer Encoder

综上给出`ConformerEncoder`的定义：

```python
class ConformerEncoder(nn.Layer):
    def __init__(
        self,
        input_dim = 80,
        encoder_dim = 512,
        num_layers = 17,
        num_attention_heads = 8,
        feed_forward_expansion_factor = 4,
        conv_expansion_factor = 2,
        input_dropout_p = 0.1,
        feed_forward_dropout_p = 0.1,
        attention_dropout_p = 0.1,
        conv_dropout_p = 0.1,
        conv_kernel_size = 31,
        half_step_residual = True
    ):
        super(ConformerEncoder, self).__init__()
        self.conv_subsample = Conv2DSubSampling(in_channels=1, out_channels=encoder_dim)
        self.input_projection = nn.Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(input_dropout_p)
        )
        self.layers = nn.LayerList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            ) for _ in range(num_layers)
        ])

    def count_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p):
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs, input_lengths):
        # inputs: (batch_size, length, input_dim)
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)    #outputs: (batch_size, sampled_length, encoder_dim)

        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs, output_lengths
```

### 2.5 Conformer实现

```python
class Conformer(nn.Layer):
    def __init__(
            self,
            num_classes,
            input_dim = 80,
            encoder_dim = 512,
            num_encoder_layers = 17,
            num_attention_heads = 8,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            input_dropout_p = 0.1,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size = 31,
            half_step_residual = True,
    ):
        super(Conformer, self).__init__()
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)
        self.log_softmax = nn.LogSoftmax(axis=-1)

    def count_parameters(self):
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p):
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs, input_lengths):
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = self.log_softmax(outputs)
        return outputs, encoder_output_lengths
```

### 2.6 其他辅助代码

此外，还有一些比较边缘但不可或缺的代码实现。这些代码在前面中已经使用，但是考虑行文结构放在最后。

```python
class Linear(nn.Layer):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0))
        if bias:
            self.linear = nn.Linear(in_features, out_features, weight_attr=weight_attr, bias_attr=bias_attr)
        else:
            self.linear = nn.Linear(in_features, out_features, weight_attr=weight_attr, bias_attr=False)
    def forward(self, x):
        return self.linear(x)

class Transpose(nn.Layer):
    def __init__(self, shape):
        super(Transpose, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.transpose(self.shape)

class Swish(nn.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        return inputs * self.sigmoid(inputs)

class GLU(nn.Layer):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        outputs, gate = inputs.chunk(chunks=2, axis=self.dim)
        return outputs * self.sigmoid(gate)
```