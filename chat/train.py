# ============================================================================
# train.py — 미니 GPT 학습 루프
#
# "학습"이란 결국 아래 4단계를 수천~수만 번 반복하는 것이다.
#   1) 데이터에서 (입력, 정답) 한 뭉치를 무작위로 뽑는다.       -> get_batch
#   2) 모델에 입력을 넣어 예측하고, 정답과 얼마나 다른지(loss)를 계산한다. -> model(x, y)
#   3) loss를 각 가중치로 미분해서(역전파, backpropagation) "어느 방향으로
#      가중치를 바꾸면 loss가 줄어드는지" 구한다.                -> loss.backward()
#   4) 그 방향으로 가중치를 아주 조금씩 이동시킨다.               -> optimizer.step()
#
# 이 과정을 통해 모델은 "다음 문자를 더 잘 맞히는" 방향으로 스스로
# 파라미터(가중치)를 조금씩 조정해 나간다.
# ============================================================================

import argparse
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_device():
    # Apple Silicon(M1/M2/...)의 GPU 가속 백엔드가 MPS.
    # 없으면 NVIDIA GPU(cuda), 그것도 없으면 CPU 순으로 fallback.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data():
    # np.memmap: 파일 전체를 메모리에 올리지 않고, 필요한 부분만 그때그때
    # 디스크에서 읽어오는 방식. train.bin이 아무리 커도 RAM을 거의 안 씀.
    train_ids = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val_ids = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return train_ids, val_ids, meta


def get_batch(data, block_size, batch_size, device):
    """전체 토큰 스트림에서 (입력 x, 정답 y) 쌍을 batch_size개 무작위로 뽑는다.

    핵심 아이디어: y는 x를 "한 칸 뒤로 민 것"이다.
      예) data = "HELLO WORLD" 이고 block_size=4, 시작 위치 i=0 이라면
          x = "HELL"  (인덱스 0~3)
          y = "ELLO"  (인덱스 1~4)
      즉 x의 각 위치 t에서, y의 같은 위치 t는 "x[t] 다음에 실제로 온 글자"가 된다.
      x[0]='H' 다음엔 y[0]='E', x[1]='E' 다음엔 y[1]='L' ... 이런 식으로
      한 시퀀스 안에서 block_size개의 "다음 글자 맞히기" 학습 샘플을 동시에 얻는 것.

    ix: batch_size개의 무작위 시작 위치. 매 스텝마다 데이터의 다른 부분을
        무작위로 뽑아 학습하므로(=확률적 경사하강, SGD), 특정 구간에 치우치지 않고
        전체 데이터를 골고루 학습하게 된다.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    """train loss와 val loss를 각각 여러 배치에 대해 평균내어 추정한다.

    한 배치만으로 loss를 보면 무작위성 때문에 값이 들쭉날쭉하므로,
    eval_iters번 반복해 평균을 내서 "지금 모델 상태"를 더 안정적으로 판단한다.

    train loss와 val loss를 같이 보는 이유(과적합 감시):
      - 둘 다 꾸준히 떨어지면 정상적으로 패턴을 배우는 중.
      - train loss는 계속 떨어지는데 val loss는 오히려 오르기 시작하면,
        모델이 학습 데이터를 "이해"하는 대신 "암기"하기 시작했다는 신호
        (=과적합, overfitting). 이 프로젝트는 val loss가 가장 낮았던 시점의
        체크포인트만 저장해서 이 문제를 자동으로 피한다.
    """
    model.eval()  # dropout 등을 비활성화 (평가 시엔 무작위성을 끔)
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()  # 다시 학습 모드로 복귀 (dropout 등 재활성화)
    return out


def main():
    parser = argparse.ArgumentParser()
    # --- 모델 크기 관련 하이퍼파라미터 (model.py의 GPTConfig와 대응) ---
    parser.add_argument("--block_size", type=int, default=128)   # 문맥 길이
    parser.add_argument("--batch_size", type=int, default=64)    # 한 스텝에 동시에 처리할 시퀀스 개수
    parser.add_argument("--n_layer", type=int, default=4)        # Transformer Block 층 수
    parser.add_argument("--n_head", type=int, default=4)         # Attention head 개수
    parser.add_argument("--n_embd", type=int, default=128)       # 임베딩 차원
    parser.add_argument("--dropout", type=float, default=0.1)
    # --- 학습 관련 하이퍼파라미터 ---
    parser.add_argument("--lr", type=float, default=3e-4)        # 학습률(learning rate):
    # 한 번의 업데이트에서 가중치를 얼마나 크게 이동시킬지. 너무 크면 loss가
    # 발산(진동/폭주)하고, 너무 작으면 학습이 지나치게 느리다. 3e-4는
    # Transformer 계열에서 경험적으로 잘 통하는 값("Karpathy constant").
    parser.add_argument("--max_iters", type=int, default=3000)   # 총 학습 스텝 수
    parser.add_argument("--eval_interval", type=int, default=250)  # 몇 스텝마다 검증할지
    parser.add_argument("--ckpt_name", type=str, default="ckpt.pt")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    train_data, val_data, meta = load_data()

    # meta["vocab_size"]는 data/prepare.py 실행 시 만들어진 실제 문자 종류 수.
    # 모델의 출력 크기(어휘 크기)가 이것과 정확히 일치해야 한다.
    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    # AdamW: 딥러닝에서 가장 널리 쓰이는 옵티마이저(가중치 업데이트 규칙) 중 하나.
    # 단순히 "gradient 반대 방향으로 lr만큼 이동"하는 기본 경사하강법(SGD)과 달리,
    # 각 파라미터마다 과거 gradient들의 평균/분산을 추적해서 파라미터별로
    # 학습 속도를 자동 조절해준다 (+ weight decay로 과적합 억제).
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(1, args.max_iters + 1):
        # --- 1. 배치 뽑기 ---
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)

        # --- 2. 순전파(forward): 예측하고 loss 계산 ---
        _, loss = model(x, y)

        # --- 3. 역전파(backward): loss를 각 가중치에 대해 미분 ---
        # zero_grad를 먼저 호출하는 이유: PyTorch는 기본적으로 gradient를
        # "누적"하므로, 매 스텝 새로 계산하려면 이전 스텝의 gradient를 지워야 함.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 이 한 줄로 모델의 모든 파라미터에 대한 gradient가 채워짐

        # --- 4. 가중치 업데이트 ---
        optimizer.step()

        # --- 주기적으로 현재 상태 점검 + 가장 좋은 모델 저장 ---
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, device)
            dt = time.time() - t0
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ({dt:.1f}s)")

            # val loss가 지금까지 중 가장 낮을 때만 저장 -> 과적합되기 전
            # "가장 일반화가 잘 된" 시점의 모델을 자동으로 보존하게 됨.
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "config": config, "meta": meta},
                    os.path.join(CKPT_DIR, args.ckpt_name),
                )

    print(f"done. best val loss: {best_val_loss:.4f}")
    print(f"checkpoint saved to {os.path.join(CKPT_DIR, args.ckpt_name)}")


if __name__ == "__main__":
    main()
