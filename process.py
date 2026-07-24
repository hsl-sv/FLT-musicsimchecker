import os
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path

def get_file_cache_key(file_path):
    """
    파일 크기(size)와 마지막 수정 시간(mtime)을 조합하여 고유 해시 생성
    (파일 전체를 읽지 않고 메타데이터만 확인하므로 속도가 극도로 빠름)
    """
    p = Path(file_path)
    stat = p.stat()
    key_str = f"{p.name}_{stat.st_size}_{stat.st_mtime}"

    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def extract_robust_features_cached(audio_path, cache_dir=".cache", target_sr=11025, hop_length=512):
    """
    캐시 파일(.npz)이 존재하면 즉시 로드하고, 없으면 피처 연산 후 저장합니다.
    """
    cache_folder = Path(cache_dir)
    cache_folder.mkdir(parents=True, exist_ok=True)
    
    # 1. 파일 고유 캐시 경로 생성 (.cache/해시값.npz)
    cache_key = get_file_cache_key(audio_path)
    cache_file = cache_folder / f"{cache_key}.npz"
    
    # 2. [캐시 HIT] 파일이 이미 존재하면 바이너리 로드 (0.01초 소요)
    if cache_file.exists():
        try:
            data = np.load(cache_file)
            return data["chroma_sync"], data["beat_times"], float(data["duration"])
        except Exception:
            # 캐시 파일이 손상되었을 경우 재계산 유도
            pass

    # 3. [캐시 MISS] 캐시가 없으면 피처 연산 수행
    chroma_sync, beat_times, duration = extract_robust_features_fast(audio_path, target_sr, hop_length)
    
    # 4. 연산 결과를 압축 바이너리로 캐시 저장
    np.savez_compressed(
        cache_file,
        chroma_sync=chroma_sync,
        beat_times=beat_times,
        duration=np.array(duration)
    )
    
    return chroma_sync, beat_times, duration

def extract_robust_features_fast(audio_path, target_sr=11025, hop_length=512):
    """
    고속 피처 추출 (Chroma CQT + L2 Normalization)
    """
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True, res_type="kaiser_fast")
    duration = len(y) / sr
    
    # 1. STFT & HPSS
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    S_harmonic, _ = librosa.decompose.hpss(S)
    
    # 2. 비트 트래킹
    onset_env = librosa.onset.onset_strength(S=S_harmonic, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    
    # 3. Chroma STFT + L2 Normalization
    chroma_raw = librosa.feature.chroma_stft(S=S_harmonic, sr=sr, hop_length=hop_length)
    chroma = librosa.util.normalize(chroma_raw, norm=2, axis=0)
    
    # 4. 비트 동기화
    if len(beat_frames) > 0:
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    else:
        chroma_sync = chroma
        beat_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
        
    return chroma_sync, beat_times, duration


def _process_single_B_worker(args):
    """
    [C#의 Parallel.ForEach 내부 람다 함수에 해당]
    각 멀티프로세스 코어에서 B 파일 1개를 독립적으로 처리하는 작업 단위
    """
    path_B, chroma_A, N_A, cost_threshold, min_coverage = args
    filename_B = os.path.basename(path_B)
    
    try:
        # 1. B 음원 피처 추출
        chroma_B, beats_B, dur_B = extract_robust_features_cached(path_B)
        N_B = chroma_B.shape[1]
        
        # 2. 12가지 피치 변조 DTW 비교 (단일 프로세스 내부 연산)
        best_cost = float('inf')
        best_wp = None
        best_shift = 0

        for shift in range(12):
            chroma_A_shifted = np.roll(chroma_A, shift, axis=0)
            D, wp = librosa.sequence.dtw(X=chroma_A_shifted, Y=chroma_B, subseq=True)
            
            if np.max(wp[:, 0]) < N_A and np.max(wp[:, 1]) < N_B:
                idx_A_col, idx_B_col = 0, 1
            else:
                idx_A_col, idx_B_col = 1, 0
                
            b_end_idx = wp[0, idx_B_col] if wp[0, idx_A_col] >= wp[-1, idx_A_col] else wp[-1, idx_B_col]
            b_end_idx = min(int(b_end_idx), N_B - 1)
            
            path_cost = D[-1, b_end_idx] / len(wp)
            
            if path_cost < best_cost:
                best_cost = path_cost
                best_wp = wp
                best_shift = shift

        # 3. 매칭 검증
        if np.max(best_wp[:, 0]) < N_A and np.max(best_wp[:, 1]) < N_B:
            b_indices = best_wp[:, 1]
        else:
            b_indices = best_wp[:, 0]
            
        start_beat_B = int(np.min(b_indices))
        end_beat_B = int(np.max(b_indices))
        
        matched_beats_B = abs(end_beat_B - start_beat_B)
        coverage_ratio = matched_beats_B / max(N_A, 1)
        slope = matched_beats_B / max(len(best_wp), 1)
        
        is_valid_cost = (best_cost <= cost_threshold)
        is_valid_coverage = (coverage_ratio >= min_coverage)
        is_valid_slope = (0.70 <= slope <= 1.45)
        
        is_match = is_valid_cost and is_valid_coverage and is_valid_slope
        
        start_time_B = beats_B[start_beat_B] if start_beat_B < len(beats_B) else dur_B
        end_time_B = beats_B[end_beat_B] if end_beat_B < len(beats_B) else dur_B
        
        start_pct_B = (start_time_B / dur_B) * 100
        end_pct_B = (end_time_B / dur_B) * 100
        pitch_change_msg = f"+{best_shift} semitones" if best_shift <= 6 else f"-{12 - best_shift} semitones"
        
        return {
            "filename_B": filename_B,
            "is_match": is_match,
            "cost": best_cost,
            "coverage_pct": coverage_ratio * 100,
            "slope": slope,
            "start_pct": start_pct_B,
            "end_pct": end_pct_B,
            "start_time": start_time_B,
            "end_time": end_time_B,
            "dur_B": dur_B,
            "pitch_msg": pitch_change_msg,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename_B": filename_B,
            "is_match": False,
            "cost": float('inf'),
            "status": f"error: {str(e)}"
        }


def analyze_1_to_n_match_parallel(path_A, paths_B, cost_threshold=0.15, min_coverage=0.50, max_workers=None):
    """
    [C#의 Parallel.ForEach 대응] 
    ProcessPoolExecutor를 활용해 N개의 B 음원 파일들을 CPU 코어별로 병렬 분산 처리합니다.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(paths_B))
        
    print("="*75)
    print(f" [병렬 분석 시작 - Parallel.ForEach] 기준 곡: {os.path.basename(path_A)}")
    print(f" 사용 CPU 코어(프로세스 수): {max_workers}개 / 비교 파일 총 {len(paths_B)}개")
    print("="*75)
    
    # 1. 기준 곡 A 피처 추출 (메인 프로세스에서 1회 수행)
    print("기준 곡(A) 피처 추출 및 캐싱 중...")
    chroma_A, beats_A, dur_A = extract_robust_features_cached(path_A)
    N_A = chroma_A.shape[1]
    
    # 2. 각 작업 프로세스에 전달할 인자 작업 튜플 생성
    tasks = [
        (path_B, chroma_A, N_A, cost_threshold, min_coverage)
        for path_B in paths_B
    ]
    
    results_list = []
    
    # 3. [C# Parallel.ForEach] 멀티프로세스 풀 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 비동기 작업 제출
        future_to_path = {executor.submit(_process_single_B_worker, task): task[0] for task in tasks}
        
        # 완료되는 순서대로 실시간 결과 수집 (Progress)
        for completed_count, future in enumerate(as_completed(future_to_path), 1):
            res = future.result()
            results_list.append(res)
            print(f"[{completed_count}/{len(paths_B)}] 완료: {res['filename_B']} (Cost: {res['cost']:.4f})")
            
    # 정렬: Match True 우선, Cost 낮은 순
    results_list.sort(key=lambda x: (not x["is_match"], x["cost"]))
            
    return results_list