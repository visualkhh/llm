# ============================================================================
# data/prepare.py — 원문 텍스트를 모델이 학습할 수 있는 "숫자 배열"로 변환
#
# 신경망은 글자를 이해하지 못하고 오직 숫자(벡터/텐서)만 다룰 수 있다.
# 그래서 텍스트를 학습시키기 전에 반드시 "토큰화(tokenization)" 과정이 필요하다.
#
# 이 프로젝트는 가장 단순한 방식인 "문자 단위(character-level)" 토큰화를 쓴다:
#   - 말뭉치(corpus)에 등장하는 모든 "고유한 문자"를 모아 사전(vocab)을 만들고,
#   - 문자 하나 <-> 정수 하나를 1:1로 대응시킨다.
#   예) "Hi" -> ['H', 'i'] -> [23, 45]  (숫자는 예시)
#
# 실제 상용 LLM(GPT 등)은 BPE(Byte-Pair Encoding) 같은 "subword" 토큰화를 써서
# 여러 글자를 묶은 조각(token) 단위로 다루지만, 문자 단위는 원리를 이해하기
# 훨씬 쉽고 구현이 단순해서 교육용으로 자주 쓰인다. (단점: 시퀀스가 길어짐)
# ============================================================================

import os
import pickle
import numpy as np

INPUT_PATH = os.path.join(os.path.dirname(__file__), "input.txt")
OUT_DIR = os.path.dirname(__file__)


def main():
    # 1) 원문 전체를 하나의 문자열로 읽어들인다.
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) 등장하는 모든 "고유 문자"를 정렬해서 사전(vocab)을 만든다.
    #    set()으로 중복 제거 -> sorted()로 순서를 고정(재현성을 위해 항상 같은 순서가 되게).
    chars = sorted(set(text))

    # stoi: string -> integer (문자를 인덱스로), itos: integer -> string (인덱스를 문자로)
    # 이 두 매핑이 곧 우리가 만든 "토크나이저"의 전부다.
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        # 문자열 -> 정수 리스트
        return [stoi[c] for c in s]

    # 3) 데이터를 학습용(train) / 검증용(val)으로 나눈다.
    #    검증셋은 학습에 전혀 쓰지 않고, "학습이 끝난 뒤 얼마나 일반화됐는지"
    #    (즉 외운 게 아니라 패턴을 배웠는지) 확인하는 용도로만 쓴다.
    n = len(text)
    train_text = text[: int(n * 0.9)]   # 앞 90%
    val_text = text[int(n * 0.9):]       # 뒤 10%

    # 4) 정수 리스트를 uint16 배열로 저장.
    #    - vocab_size가 보통 수십~수백 수준이라 uint16(0~65535) 범위로 충분하고,
    #      uint32보다 파일 용량이 절반이라 디스크 I/O에 유리하다.
    train_ids = np.array(encode(train_text), dtype=np.uint16)
    val_ids = np.array(encode(val_text), dtype=np.uint16)

    # .bin 파일로 저장해두면 train.py에서 np.memmap으로 "디스크에 둔 채로"
    # 필요한 부분만 읽어올 수 있어, 말뭉치가 아무리 커도 메모리를 적게 쓴다.
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    # 5) 나중에 모델을 불러와 생성(chat.py)할 때도 똑같은 stoi/itos가 필요하므로
    #    meta.pkl에 함께 저장해둔다. (train.py가 저장하는 체크포인트에도 다시 포함됨)
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)

    print(f"vocab size: {len(chars)}")
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    main()
