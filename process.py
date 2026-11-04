import os
import hashlib
import json
import tempfile
import warnings
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import librosa
import numpy as np
from scipy.spatial.distance import cdist
from numba import njit
from threadpoolctl import threadpool_limits


def get_file_cache_key(file_path):
    p = Path(file_path).resolve()
    stat = p.stat()
    key_str = f'{p}_{stat.st_size}_{stat.st_mtime_ns}'
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()


def _feature_cache_key(audio_path, target_sr, hop_length):
    payload = [get_file_cache_key(audio_path), target_sr, hop_length,
               'qt-compatible-v1', librosa.__version__, np.__version__]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def extract_robust_features_cached(audio_path, cache_dir='.cache', target_sr=11025, hop_length=512):
    """기존과 동일하게 (chroma_sync, beat_times, duration) 반환."""
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    key = _feature_cache_key(audio_path, target_sr, hop_length)
    dest = folder / f'{key}.npz'
    if dest.exists():
        try:
            with np.load(dest, allow_pickle=False) as d:
                x, times, duration = d['chroma_sync'], d['beat_times'], float(d['duration'])
            if (x.ndim != 2 or x.shape[0] != 12 or x.shape[1] == 0 or
                times.ndim != 1 or not np.isfinite(x).all() or
                not np.isfinite(times).all() or not np.isfinite(duration) or duration <= 0):
                raise ValueError('잘못된 캐시')
            return x, times, duration
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass
    result = extract_robust_features_fast(audio_path, target_sr, hop_length)
    if key != _feature_cache_key(audio_path, target_sr, hop_length):
        raise RuntimeError('분석 중 음원 파일이 변경되었습니다')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as f:
            tmp = Path(f.name)
            np.savez(f, chroma_sync=result[0], beat_times=result[1], duration=np.array(result[2]))
        os.replace(tmp, dest)
    except OSError as e:
        warnings.warn(f'캐시 저장 실패, 계산 결과는 사용: {e}')
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return result


def extract_robust_features_fast(audio_path, target_sr=11025, hop_length=512):
    """원본 피처 및 beat_times 계산을 유지하여 기존 Qt 코드와 호환."""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True, res_type='kaiser_fast')
    duration = len(y) / sr
    if duration <= 0 or not np.isfinite(y).all():
        raise ValueError('빈 음원 또는 비정상 샘플')
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    H, _ = librosa.decompose.hpss(S)
    onset = librosa.onset.onset_strength(S=H, sr=sr, hop_length=hop_length)
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop_length)
    raw = librosa.feature.chroma_stft(S=H, sr=sr, hop_length=hop_length)
    chroma = librosa.util.normalize(raw, norm=2, axis=0)
    if len(beats):
        synced = librosa.util.sync(chroma, beats, aggregate=np.median)
        times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    else:
        synced = chroma
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    return synced, times, duration

@njit(cache=True, nogil=True)
def _dtw_summary(C):
    """기본 steps=(1,1),(0,1),(1,0), 유클리드 비음수 비용 전용.
    librosa.dtw(C=C, subseq=True)의 최저 누적 비용 끝점과 같은 선택.
    평균 비용 자체를 전역 최적화하는 알고리즘은 아님.
    C 외 DP 메모리는 O(N_B); 전체 wp는 반환하지 않음.
    """
    n, m = C.shape
    prev = C[0].copy()
    curr = np.empty(m, np.float64)
    lengths = np.ones(m, np.int64)
    next_lengths = np.empty(m, np.int64)
    starts = np.arange(m)
    next_starts = np.empty(m, np.int64)
    for i in range(1, n):
        curr[0] = prev[0] + C[i, 0]
        next_lengths[0] = lengths[0] + 1
        next_starts[0] = starts[0]
        for j in range(1, m):
            # 비교는 합산 후: librosa와 동일한 동률 우선순위.
            best = prev[j-1] + C[i, j]
            length = lengths[j-1]
            start = starts[j-1]
            left = curr[j-1] + C[i, j]
            if left < best:
                best, length, start = left, next_lengths[j-1], next_starts[j-1]
            up = prev[j] + C[i, j]
            if up < best:
                best, length, start = up, lengths[j], starts[j]
            curr[j], next_lengths[j], next_starts[j] = best, length+1, start
        prev, curr = curr, prev
        lengths, next_lengths = next_lengths, lengths
        starts, next_starts = next_starts, starts
    end = np.argmin(prev)
    return prev[end] / lengths[end], starts[end], end, lengths[end]


def _prepare_shifts(chroma_A):
    return tuple(np.ascontiguousarray(np.roll(chroma_A, s, axis=0).T, dtype=np.float64)
                 for s in range(12))


def _compare_single_B(path_B, a_shifts, N_A, cost_threshold, min_coverage):
    filename_B = os.path.basename(path_B)
    try:
        chroma_B, beats_B, dur_B = extract_robust_features_cached(path_B)
        N_B = chroma_B.shape[1]
        b = np.ascontiguousarray(chroma_B.T, dtype=np.float64)
        costs = np.empty((N_A, N_B), dtype=np.float64)
        best = None
        for shift, a in enumerate(a_shifts):
            if a.shape != (N_A, 12):
                raise ValueError('N_A와 chroma_A 크기가 다릅니다')
            cdist(a, b, metric='euclidean', out=costs)
            if N_A <= N_B:
                score, begin, end, length = _dtw_summary(costs)
            else:
                # 기존 librosa X/Y 호출처럼 짧은 B 전체를 A 내부에 매칭.
                # 원본의 D[-1, b_end_idx] 오인덱싱 대신 실제 끝점 비용 사용.
                score, _, _, length = _dtw_summary(np.ascontiguousarray(costs.T))
                begin, end = 0, N_B - 1
            if best is None or score < best[0]:
                best = (score, begin, end, length, shift)
        best_cost, start_beat_B, end_beat_B, path_length, best_shift = best
        matched_beats_B = end_beat_B - start_beat_B
        coverage_ratio = matched_beats_B / max(N_A, 1)
        slope = matched_beats_B / max(path_length, 1)
        is_match = (best_cost <= cost_threshold and coverage_ratio >= min_coverage
                    and 0.70 <= slope <= 1.45)
        # 외부 시간 출력도 기존 기준을 유지.
        start_time_B = beats_B[start_beat_B] if start_beat_B < len(beats_B) else dur_B
        end_time_B = beats_B[end_beat_B] if end_beat_B < len(beats_B) else dur_B
        pitch_change_msg = f'+{best_shift} semitones' if best_shift <= 6 else f'-{12-best_shift} semitones'
        return {
            'filename_B': filename_B,
            'is_match': bool(is_match),
            'cost': float(best_cost),
            'coverage_pct': coverage_ratio * 100,
            'slope': slope,
            'start_pct': (start_time_B / dur_B) * 100,
            'end_pct': (end_time_B / dur_B) * 100,
            'start_time': start_time_B,
            'end_time': end_time_B,
            'dur_B': dur_B,
            'pitch_msg': pitch_change_msg,
            'status': 'success',
        }
    except Exception as e:
        return {'filename_B': filename_B, 'is_match': False,
                'cost': float('inf'), 'status': f'error: {e}'}


def _process_single_B_worker(args):
    """기존 튜플 (path_B, chroma_A, N_A, cost_threshold, min_coverage) 호환."""
    path_B, chroma_A, N_A, cost_threshold, min_coverage = args
    try:
        return _compare_single_B(path_B, _prepare_shifts(chroma_A), N_A,
                                 cost_threshold, min_coverage)
    except Exception as e:
        return {'filename_B': os.path.basename(path_B), 'is_match': False,
                'cost': float('inf'), 'status': f'error: {e}'}


def analyze_1_to_n_match_parallel(path_A, paths_B, cost_threshold=0.15, min_coverage=0.50, max_workers=None):
    """동기 함수: Qt의 기존 QThread/worker에서 호출. 결과 list[dict] 반환.
    내부 ThreadPool + nogil DTW로 병렬 처리하며 main/프로세스 초기화 불필요.
    """
    paths_B = list(paths_B)
    if not paths_B:
        return []
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(paths_B))
    if max_workers < 1:
        raise ValueError('max_workers는 1 이상이어야 합니다')
    print('=' * 75)
    print(f' [병렬 분석 시작 - Parallel.ForEach] 기준 곡: {os.path.basename(path_A)}')
    print(f' 사용 작업 스레드: {max_workers}개 / 비교 파일 총 {len(paths_B)}개')
    print('=' * 75)
    print('기준 곡(A) 피처 추출 및 캐싱 중...')
    results_list = []
    # BLAS 내부 스레드와 작업 스레드의 과도한 중첩 방지.
    with threadpool_limits(limits=1):
        chroma_A, beats_A, dur_A = extract_robust_features_cached(path_A)
        N_A = chroma_A.shape[1]
        a_shifts = _prepare_shifts(chroma_A)
        # 작업 시작 전 JIT 준비. 이후 핵심 DP는 GIL 없이 실행.
        _dtw_summary(np.zeros((2, 2), dtype=np.float64))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compare_single_B, p, a_shifts, N_A,
                                       cost_threshold, min_coverage): p for p in paths_B}
            for completed_count, future in enumerate(as_completed(futures), 1):
                try:
                    res = future.result()
                except Exception as e:
                    res = {'filename_B': os.path.basename(futures[future]),
                           'is_match': False, 'cost': float('inf'), 'status': f'error: {e}'}
                results_list.append(res)
                print(f"[{completed_count}/{len(paths_B)}] 완료: {res['filename_B']} (Cost: {res['cost']:.4f})")
    results_list.sort(key=lambda x: (not x['is_match'], x['cost']))
    return results_list
