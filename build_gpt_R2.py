# part 1: 导入相关的 package
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

import math

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768  # 确保 n_embd 是 n_head 的整数倍
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257

class SingleHeadAttention(nn.Module):
    # 单头注意力机制
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size

        # 尝试学习新的写法，attention_mask 通过 register_buffer 注册
        # 因为不用计算 梯度，所以节约内存和显存，速度也更快
        self.register_buffer(
            'attention_mask', 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)   # @ 就是 torch.matmul 的简化写法
        # 一定要在 softmax 前除以 sqrt(head_size)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, 
            float('-inf')
        ) / math.sqrt(self.head_size)  # 这里的 hidden_size 其实是 head_size，因为是单头
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs.sin(), freqs.cos()

class GroupedQueryAttentionWithRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_q_heads = config.n_head  # e.g., 12
        self.n_kv_heads = config.n_head // 2  # e.g., 6
        assert config.n_head % 2 == 0, "n_head must be divisible by 2 for GQA"

        self.head_size = config.n_embd // config.n_head
        assert self.head_size % 2 == 0, "head_size must be even for RoPE"

        # Query, Key, Value projection
        self.q_proj = nn.Linear(config.n_embd, self.n_q_heads * self.head_size)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.rotary_emb = RotaryEmbedding(self.head_size)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "attention_mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, x, sin, cos):
        # x: (batch, n_head, seq_len, head_size)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,head_size)
        cos = cos.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, x):
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_size).transpose(1, 2)  # (B, nq, T, H)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)  # (B, nkv, T, H)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)  # (B, nkv, T, H)

        sin, cos = self.rotary_emb(T, x.device)  # (T, H)
        q = self.apply_rotary(q, sin, cos)
        k = self.apply_rotary(k, sin, cos)

        # Expand kv heads for broadcasting (B, nkv, T, H) -> (B, nq, T, H)
        if self.n_q_heads != self.n_kv_heads:
            expand_factor = self.n_q_heads // self.n_kv_heads
            k = k.repeat_interleave(expand_factor, dim=1)
            v = v.repeat_interleave(expand_factor, dim=1)

        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)  # (B, nq, T, T)
        att = att.masked_fill(self.attention_mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # (B, nq, T, H)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

# 接下来就是一个完整的 Block

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GroupedQueryAttentionWithRoPE(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.ln2 = nn.RMSNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 以后会讲  MLA ,  MOE, DPO 完全手写
# 完整的  GPT model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.token_embedding_table(idx)
        x = token_emb

        for i, block in enumerate(self.blocks):
            x = block(x)
            print(f"Block {i} output shape: {x.shape}")  # 调试信息

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            logits_flat = logits.view(-1, logits.size(-1))       # [batch*seq_len, vocab_size]
            targets_flat = targets.view(-1)                      # [batch*seq_len]
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 如果序列太长，只取最后 block_size 个token
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 获取预测
            logits, _ = self(idx_cond)
            # 只关注最后一个时间步的预测
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx



# 写一个 dataset，为了 Dataloader 准备
class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        # 我的数据在 /root/fs/mobvoi_seq_monkey_general_open_corpus.jsonl 中，
        # 读取前 1000 行
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

        self.encoded_data = []

        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)


# train data
train_dataset = MyDataset('../fs/mobvoi_seq_monkey_general_open_corpus.jsonl')

# split traindataset to train and val
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

model = GPT(GPTConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 打印模型一共有多少参数

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# 设置 cosine 学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# 训练循环
def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, targets=y)
        if logits.size(1) != y.size(1):
            raise RuntimeError(f"Mismatch in sequence length: logits.size(1)={logits.size(1)}, y.size(1)={y.size(1)}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    perplexity = torch.exp(torch.tensor(val_loss / len(val_loader)))
    print(f'Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}')
    return val_loss


for epoch in range(2):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
    val_loss = eval(model, val_loader, device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    # 保存模型
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }
    torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
    