# ============================================================================
# chat.py — 학습된 체크포인트를 불러와 텍스트를 생성하는 간단한 REPL
#
# 주의: 이 모델은 "질문에 답하도록" 학습된 것이 아니라, 그냥 셰익스피어
# 텍스트 "다음에 올 글자"를 이어가도록만 학습되었다. 그래서 여기서 하는
# "대화"란 사실 사용자가 입력한 문자열을 프롬프트(시작 문장)로 주고,
# 모델이 그 뒤를 계속 이어 쓰게 하는 것에 가깝다. (진짜 질의응답 챗봇을
# 만들려면 질문-답변 쌍으로 된 데이터로 별도 미세조정이 필요함)
# ============================================================================

import argparse
import os

import torch

from model import GPT, GPTConfig

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(ckpt_name, device):
    # train.py가 저장한 체크포인트에는 세 가지가 함께 들어있다:
    #   "model"  : 학습된 가중치 (state_dict)
    #   "config" : 그 가중치가 어떤 GPTConfig(모델 크기)로 학습됐는지
    #   "meta"   : 학습에 쓰인 stoi/itos 문자 매핑
    # 이 세 가지가 항상 함께 저장/로드되어야 하는 이유: 가중치의 각 텐서
    # 크기는 config(n_embd, n_layer 등)에 딱 맞게 만들어져 있고, 가중치가
    # "학습한 대상"인 문자 매핑도 meta와 정확히 일치해야 하기 때문이다.
    ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # dropout 비활성화: 추론(생성) 시에는 항상 결정적인 forward가 되어야 함
    return model, ckpt["meta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_name", type=str, default="ckpt.pt")
    parser.add_argument("--max_new_tokens", type=int, default=300)   # 한 번에 몇 글자 생성할지
    parser.add_argument("--temperature", type=float, default=0.8)    # 무작위성 정도 (model.py의 generate 참고)
    parser.add_argument("--top_k", type=int, default=50)             # 후보를 상위 몇 개로 제한할지
    args = parser.parse_args()

    device = get_device()
    model, meta = load_model(args.ckpt_name, device)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        # 학습 때 못 본 문자(stoi에 없는 문자)는 조용히 무시한다.
        # (예: 학습 말뭉치에 없던 한글/이모지 등을 입력하면 그 글자는 그냥 사라짐)
        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    print("Mini-GPT chat. Type a prompt and press Enter. Ctrl+C to quit.\n")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue

        # 사용자의 프롬프트를 인덱스로 바꾸고, batch 차원을 하나 추가해 (1, T) 형태로 만든다.
        # (모델은 항상 (B, T) 형태의 배치 입력을 기대하므로 B=1이라도 차원을 맞춰야 함)
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        # 프롬프트 뒤에 max_new_tokens개의 글자를 자기회귀적으로 이어 생성
        out = model.generate(
            idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
        )
        text = decode(out[0].tolist())

        # out에는 "프롬프트 + 새로 생성된 부분"이 모두 들어있으므로,
        # 프롬프트 길이만큼 앞을 잘라내 새로 생성된 부분만 출력한다.
        print(f"gpt> {text[len(prompt):]}\n")


if __name__ == "__main__":
    main()
