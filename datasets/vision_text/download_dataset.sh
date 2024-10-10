#!/bin/bash

# 로그 파일 설정
LOG_FILE="dataset_download.log"

# 로그 함수
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 오류 발생 시 스크립트 중단
set -e

# 시작 시간 기록
START_TIME=$(date +%s)

log "데이터셋 다운로드 시작"

# PubMedVision 데이터셋 다운로드
log "PubMedVision 데이터셋 다운로드 시작"
python down_FreedomIntelligencePubMedVision.py
log "PubMedVision 데이터셋 다운로드 완료"

# MEDPIX-ClinQA 데이터셋 다운로드
log "MEDPIX-ClinQA 데이터셋 다운로드 시작"
python down_adishouryaMEDPIX-ClinQA.py
log "MEDPIX-ClinQA 데이터셋 다운로드 완료"

# Path-VQA 데이터셋 다운로드
log "Path-VQA 데이터셋 다운로드 시작"
python down_flaviagiammarinopath-vqa.py
log "Path-VQA 데이터셋 다운로드 완료"

# VQA-RAD 데이터셋 다운로드
log "VQA-RAD 데이터셋 다운로드 시작"
python down_flaviagiammarinovqa-rad.py
log "VQA-RAD 데이터셋 다운로드 완료"

# 종료 시간 기록 및 총 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "모든 데이터셋 다운로드 완료"
log "총 소요 시간: $(($DURATION / 3600))시간 $(($DURATION % 3600 / 60))분 $(($DURATION % 60))초"
