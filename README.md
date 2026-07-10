# QMR: Q-learning 기반 UAV 애드혹 네트워크 라우팅

졸업논문/캡스톤 프로젝트 (3인 팀)

> ⚠️ **이 레포는 논문 전체가 아니라, 본인이 맡았던 "시뮬레이션 성능 검증 + 결과
> 시각화" 파트를 정리한 것입니다.** 알고리즘 이론 설계와 구현은 팀원과 함께
> 진행했으며, 여기서는 검증 과정에서 본인이 직접 다룬 실험과 발견한 내용을
> 중심으로 서술합니다.

## 기반 및 출처

- **시뮬레이터**: [DroNet](https://github.com/Andrea94c/DroNETworkSimulator) —
  Andrea Coletta, Matteo Prata (Sapienza University of Rome)
- **알고리즘**: Liu, J. et al. (2020). QMR: Q-learning based Multi-objective
  optimization Routing protocol for Flying Ad Hoc Networks. *Computer
  Communications*, 150, 304–316. https://doi.org/10.1016/j.comcom.2019.11.011
- 라이선스: GPL v3 (원본 유지)

팀이 이 위에 직접 추가한 것: `alg_suffix` 기반 파라미터 주입 프레임워크,
실험 시나리오 설계, **Deadline-Aware Two-Mode Policy** 확장.

## 실험 1. β(MAC 혼잡 계수) 조정

- **예상**: 고밀도/혼잡 환경에서 혼잡 회피를 강화(β 0.5→0.8)하면 전달률이 오를 것
- **실제 결과**: 20 seed 재검증 결과 오히려 소폭 하락 (0.770 → 0.724)
- **해결방안**: 단순 파라미터 조정만으로는 개선이 어렵다고 판단, 의사결정 구조
  자체를 바꾸는 실험 2로 이어짐

![delivery ratio](corrected_s3_delivery_ratio.png)

## 실험 2. Deadline-Aware Two-Mode Policy

- **예상**: 마감(TTL)이 임박한 패킷만 목적지 방향으로 강제 수렴시키면 최악의
  지연 사례가 줄어들 것
- **실제 결과**: 최악 지연(max)·상위 5% 지연(p95) 모두 일관되게 감소
  (예: max 584s → 578s). 다만 신뢰구간이 겹쳐 통계적으로는 조심스러움
- **해결방안**: seed 수를 늘려 추가 검증 필요. 페널티(DM)를 같이 켰을 때 효과가
  안 보이는 문제는 원인을 찾아 별도 기록 → [NOTES.md](NOTES.md) 참고

## 한계

- 단순 보상함수 조정(β, ω)만으로는 뚜렷한 개선이 어려웠음 (ω 실험은 결과가
  나빠서 최종 보고서에서도 제외)
- Two-Mode Policy도 신뢰구간 겹침으로 추가 검증이 필요한 상태

## 검증 과정에서 발견한 것

포트폴리오 정리 중 실험 데이터를 재검증하다가 발견한 사항들(비교 대상 데이터
불일치, 코드 레벨 원인 분석 등)은 [NOTES.md](NOTES.md)에 정리했습니다.

## 기술 스택

Python, pygame(시각화), matplotlib, numpy

## 원본 논문

Liu, J. et al. (2020). *Computer Communications*, 150, 304–316.
(저작권 문제로 PDF는 첨부하지 않으며, DOI 링크로 원문을 확인할 수 있습니다.)
