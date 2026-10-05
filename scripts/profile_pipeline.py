import time
import argparse
import psutil
import pynvml
import threading
from pathlib import Path
from queue import Empty
import numpy as np

from backend.app.core.config import load_config
from backend.app.inference import UltralyticsPersonDetector
from backend.app.tracking.bytetrack_tracker import ByteTrackTracker
from backend.app.video.pipeline import TrackingPipeline
from backend.app.video.source import OpenCVVideoSource
from backend.app.schemas import FrameResult

gpu_samples = []
cpu_samples = []
profiler_running = True

def profiler_thread():
    proc = psutil.Process()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        handle = None
    while profiler_running:
        try:
            if handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_samples.append({
                    "gpu_util": util.gpu,
                    "mem_used_mb": mem.used / (1024**2)
                })
        except Exception:
            pass
        cpu_samples.append(proc.cpu_percent())
        time.sleep(0.1)

def percentile(vals, p):
    if not vals:
        return 0.0
    return np.percentile(vals, p)

def main():
    global profiler_running
    config_path = "configs/shibuya.yaml"
    video_path = "data/videos/shibuya-20s.mp4"
    
    config = load_config(config_path)
    # Ensure it's original baseline config
    config.detector.tiled_inference = True
    config.detector.perspective_regions.enabled = False
    config.video.inference_interval = 1
    
    source = OpenCVVideoSource(video_path)
    detector = UltralyticsPersonDetector(config.detector)
    tracker = ByteTrackTracker(config.tracker)
    
    pipeline = TrackingPipeline(
        source=source,
        detector=detector,
        tracker=tracker,
        inference_interval=1,
        queue_size=4,
        drop_oldest=False
    )
    
    # We will hook into detector to extract stats per frame
    metrics_records = []
    
    def on_result(result: FrameResult):
        stats = detector.last_inference_stats
        metrics_records.append({
            "decode_ms": result.packet.decode_ms,
            "preprocess_yolo_ms": stats.get("yolo_preprocess_ms", 0),
            "infer_yolo_ms": stats.get("yolo_inference_ms", 0),
            "postprocess_yolo_ms": stats.get("yolo_postprocess_ms", 0),
            "result_transfer_ms": stats.get("result_transfer_ms", 0),
            "merge_ms": stats.get("merge_ms", 0),
            "tracking_ms": result.tracking_ms,
            "e2e_ms": result.e2e_latency_ms,
            "queue_idle_ms": result.e2e_latency_ms - (result.packet.decode_ms + result.inference_ms + result.tracking_ms)
        })
        if len(metrics_records) % 10 == 0:
            print(f"Processed {len(metrics_records)} frames...")
        if len(metrics_records) >= 30: # profile 30 frames for fast testing
            print("="*60)
            print("4. Stage timing p50/p95")
            stages = ["decode_ms", "preprocess_yolo_ms", "infer_yolo_ms", "postprocess_yolo_ms", "result_transfer_ms", "merge_ms", "tracking_ms", "queue_idle_ms", "e2e_ms"]
            stage_stats = {}
            for stage in stages:
                vals = [r[stage] for r in metrics_records]
                p50 = percentile(vals, 50)
                p95 = percentile(vals, 95)
                stage_stats[stage] = p95
                print(f"{stage:20s}: p50={p50:6.1f}ms  p95={p95:6.1f}ms")
            
            print("\n6. Top 3 bottlenecks")
            sorted_stages = sorted([(k, v) for k, v in stage_stats.items() if k not in ["e2e_ms", "queue_idle_ms"]], key=lambda x: x[1], reverse=True)
            for i, (k, v) in enumerate(sorted_stages[:3]):
                print(f"  #{i+1}: {k} ({v:.1f}ms p95)")
            print("="*60)
            import os; os._exit(0)
            
    pipeline.on_result = on_result
    
    t = threading.Thread(target=profiler_thread)
    t.start()
    
    start_time = time.perf_counter()
    pipeline.start()
    
    while pipeline._processing_thread.is_alive() and not pipeline._stop.is_set():
        time.sleep(0.5)
        
    pipeline.join()
    profiler_running = False
    t.join()
    
    elapsed = time.perf_counter() - start_time
    frames = len(metrics_records)
    
    with open("profile_report.txt", "w", encoding="utf-8") as f:
        pass # write empty to ensure it exists
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        with open("profile_error.txt", "w") as f:
            traceback.print_exc(file=f)
        raise
