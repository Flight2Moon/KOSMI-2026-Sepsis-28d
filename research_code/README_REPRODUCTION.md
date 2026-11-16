# Donga - Sepsis / Study28 EXP123 — 재현 패키지

MIMIC-IV 개발 코호트로 학습하고 eICU-CRD(2개 코호트 정의), SICdb, NWICU로
외부 검증한 **28일 3-분류 재원 결과 예측** 연구의 전체 코드·데이터·결과·산출물
패키지입니다.

이 패키지의 핵심 분석 결과는 **빌드 시점에 실제로 재실행하여 수치 일치를
확인**했습니다. 결과는 `analysis/reference/`에 동결되어 있고,
`analysis/verify_reproduction.sh`로 언제든 재검증할 수 있습니다.

---

## 1. 빠른 시작

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 재현 검증 (데이터 체크섬 -> 부트스트랩 -> 그림 -> 대조)
bash analysis/verify_reproduction.sh
```

마지막에 `[OK] Reproduction verified.`가 출력되면 패키지가 정상입니다.
소요 시간은 수 분이며, 모델 재학습은 일어나지 않습니다.

---

## 2. 디렉터리 구조

```
.
├── README.md                      원 프로젝트 문서 (EXP123 파이프라인 전반)
├── README_REPRODUCTION.md         이 문서
├── exp123                         통합 CLI 진입점
├── run_study28_full_pipeline.sh   alpha2~6 전체 파이프라인
├── run_exp123.py                  EXP123 메인 러너
│
├── src/los_baseline/              핵심 패키지 (116 파일 / 35,618 LOC)
│   ├── study28/                   Study28 지원 모듈 29개
│   ├── study28_exp123/            현재 논문 실험 본체 (core.py, hpo.py)
│   ├── hpo/                       Optuna 탐색
│   └── imbalance/                 언더샘플링 / SMOTE
│
├── tools/                         단발 유틸 19개
│   └── run_study28_exp123_two_stage.py   EXP1/2/3 비교 본체
├── scripts/workflows/             V8 트랙 매트릭스 16개
├── configs/                       실험 설정 26개
├── tests/                         pytest 스위트
├── legacy_reference/              초기 baseline 5종 (main_dt.py 등)
│
├── data/                          동결 데이터셋 4종 + feature_list
├── runs/study28_exp123_main/      학습 산출물
│   ├── stage1_baseline/           기본 설정 baseline
│   └── stage2_optimization/       Optuna 최적화 + 동결 모델
│
├── analysis/                      본 재현 패키지에서 추가된 분석
│   ├── paired_bootstrap_flat_vs_cascade.py
│   ├── make_panel_a.py
│   ├── verify_reproduction.sh
│   ├── results/                   재실행 산출물
│   ├── reference/                 동결 기준값 (대조용)
│   └── figures/                   Figure 1 (PDF/PNG)
│
└── deliverables/                  문서 산출물
    ├── 1_Experiments_Design_Review.pdf
    ├── 2_Dataset_Specification.pdf
    ├── 3_Fix_and_Optimization_Plan.pdf
    └── KOSMI2026_포스터_*.docx
```

---

## 3. 데이터셋

`configs/dataset_registry_28d.json`의 `expected_sha256`과 `data/` 안의 파일이
일치함을 빌드 시점에 확인했습니다.

| 파일 | 역할 | 행 수 | 코호트 필터 |
|---|---|---|---|
| `mimic_final.csv` | 개발 (development) | 22,741 | `anchor_year_group`으로 2008–2016 / 2017–2022 분할 |
| `eicu_final.csv` | 외부 검증 | 14,196 | `sepsis_admitdx==1` (alt) / `sepsis3_primary==1` (primary) |
| `sicdb_final.csv` | 외부 검증 | 9,122 | 없음 |
| `nwicu_final.csv` | 외부 검증 | 3,829 | 없음 |

eICU의 두 코호트는 **1,369명이 중복**되므로 독립적인 두 외부 검증이 아닙니다.
논문·초록에서 반드시 명시해야 합니다.

`all_db_stacked.csv`는 4개 파일의 통합본으로, 개별 파일과 비트 단위로
동일함을 확인했습니다. 학습·평가에는 개별 파일을 쓰십시오 (외부 데이터가
실수로 학습에 섞이는 사고 방지).

---

## 4. 분석 재현: 다중분류 vs 사망 우선 2단계

### 무엇을 하는가

`analysis/paired_bootstrap_flat_vs_cascade.py`는 **모델을 재학습하지 않습니다.**
`runs/.../stage2_optimization/models/` 아래 동결된 joblib 모델과 전처리기를
그대로 읽어 재예측만 수행합니다. 따라서 결정적(deterministic)입니다.

2단계 모델의 3-분류 확률 결합 규칙:

```
P(death)     = P1
P(early)     = (1 - P1) * (1 - P2)
P(prolonged) = (1 - P1) * P2
```

신뢰구간은 **환자 단위 짝지은 부트스트랩** 1,000회(seed 42)로 산출합니다.
두 구조를 동일한 재표본 환자에서 평가하므로 모델 간 상관이 보존되고,
차이의 신뢰구간이 좁게 나옵니다.

### 실행

```bash
python analysis/paired_bootstrap_flat_vs_cascade.py                # 5개 코호트 전부
python analysis/paired_bootstrap_flat_vs_cascade.py sicdb nwicu    # 일부만
python analysis/paired_bootstrap_flat_vs_cascade.py --merge        # 부분 결과 병합
```

### 그림

```bash
python analysis/make_panel_a.py                                 # 160 x 74 mm (포스터 본문 폭)
python analysis/make_panel_a.py --width-mm 100 --aspect 0.98    # 정사각형
python analysis/make_panel_a.py --model lightgbm                # 다른 알고리즘
python analysis/make_panel_a.py --delta-column                  # 차이 주석 열 추가
```

`--width-mm`로 지정한 값이 **인쇄 폭 그대로** 나옵니다(캔버스 크기로 저장).
160mm는 Table 1의 폭과 맞춘 값입니다.

---

## 5. 재현 검증 결과

빌드 시점에 `analysis/verify_reproduction.sh`를 실행한 결과입니다.

```
데이터 체크섬      4/4 일치
비교 행            25 (5 알고리즘 x 5 평가 세팅)
비교 수치 컬럼     14
최대 절대 차이     0.000e+00
Figure 1 픽셀 차이 0 (3779 x 1738 완전 동일)
```

주목할 점은 **라이브러리 버전이 `requirements.txt`와 다른 환경에서도 수치가
완전히 일치**했다는 것입니다.

| 라이브러리 | requirements.txt | 검증 환경 |
|---|---|---|
| pandas | 2.2.3 | 3.0.5 |
| scikit-learn | 1.8.0 | 1.9.1 |
| numpy | 2.3.5 | 2.5.3 |
| scipy | 1.17.0 | 1.18.1 |
| xgboost | 3.1.3 | 3.1.3 |
| lightgbm | 4.6.0 | 4.6.0 |

전체 환경 기록은 `analysis/reference/verified_environment.txt`에 있습니다.

### 알려진 환경 이슈 2건 (코드에 이미 반영됨)

**(1) pandas 3.x에서 feature name 해석 실패**

XGBoost / LightGBM sklearn 래퍼가 DataFrame의 컬럼명을 booster에 저장된
이름과 대조하지 못해 `data did not contain feature names` 오류가 납니다.
native-missing 모델은 manifest의 컬럼 순서대로 NumPy 배열로 변환해 넘기고
이름 검증을 끕니다. 컬럼 순서가 학습 시와 동일하므로 입력 행렬은 같습니다.

**(2) `roc_auc_score`가 부트스트랩 루프에서 지나치게 느림**

순위 기반 등가식(`scipy.stats.rankdata`)으로 대체했습니다. 동점 포함하여
scikit-learn과 1e-9까지 일치함을 확인했습니다. 이 대체 없이는 단일 코어에서
수 시간이 걸립니다.

두 조치 모두 `requirements.txt`의 고정 버전 환경에서는 동작에 영향이 없습니다.

---

## 6. 전체 실험 재실행 (선택)

위 분석은 동결 모델을 재사용하므로 빠릅니다. 모델 자체를 처음부터 다시
학습하려면 아래를 쓰되, **Optuna 최적화는 수일이 걸립니다.**

```bash
./exp123 study28-status        # 준비 상태 확인
./exp123 study28-baseline      # 기본 설정 baseline (빠름)
./exp123 study28-optimize      # Optuna 최적화 (매우 오래 걸림)
./exp123 study28-all           # baseline 후 최적화까지 연속 실행
```

SVM은 단일 fit에 약 51,000초가 관측된 적이 있습니다. 전체 탐색 전에
`scripts/workflows/benchmark_exp123_v8_svm.py`로 단일 fit 시간을 먼저
측정하십시오.

---

## 7. 미해결 결정 사항

아래는 교수님 확인이 필요하며, 확정 전에는 의존하는 코드를 작성·변경하지
않아야 합니다. 상세는 `deliverables/1_Experiments_Design_Review.pdf` 7장
참조.

| ID | 항목 | 비고 |
|---|---|---|
| D1 | time-zero 정의 (ICU 입실 vs 병원 입원) | 모든 시간 윈도우와 LOS 계산에 영향 |
| D3 | SMOTE의 Stage1 적용 여부 | 사망 유병률 왜곡 우려 |
| D4 | LOS14에서의 SMOTE 2배 적용 | LOS14는 이미 Class C가 약 24% |
| D6 | 변수 세트 확장 범위 | 목표 57변수 |
| D7 | Primary endpoint 사전 지정 | 다중비교 통제 |

---

## 8. 용어 주의

* **"Track 3"은 쓰지 마십시오.** 언더샘플링 전략 트랙과 Experiment 3이
  모두 이 이름으로 불려 혼동됩니다. `undersample-track3` / `EXP3`로
  명시하십시오.
* **EXP3의 양성 클래스는 Class C**(28일 시점 재원 중)입니다.
* 그림·표의 모델 명칭은 **Multiclass model** / **Death-first two-stage
  model**로 통일되어 있습니다. 본문도 같은 용어를 쓰십시오.
* **"동등하다(equivalent)"는 표현은 쓰지 마십시오.** 동등성 마진을 사전에
  정하지 않았으므로 "일관된 이점을 제공하지 않았다(did not provide a
  consistent advantage)"가 정확한 표현입니다.
