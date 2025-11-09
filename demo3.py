import os
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import whisper

# ---------------- utilities ----------------

def run(cmd: List[str], check: bool = True):
    try:
        subprocess.run(cmd, check=check)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}") from e

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path]
    ).decode("utf-8","ignore").strip()
    try: return float(out)
    except: return 0.0

# ---------------- audio ----------------

class AudioExtractor:
    def extract(self, video_path: str, out_wav: str) -> str:
        ensure_dir(os.path.dirname(out_wav))
        run(["ffmpeg","-y","-i",video_path,"-vn","-acodec","pcm_s16le","-ar","16000","-ac","1",out_wav])
        return out_wav

# ---------------- ASR (English) ----------------

class SpeechRecognizer:
    def __init__(self, model_size="medium", device: Optional[str]=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Whisper device: {self.device}")

    def transcribe(self, audio_path: str) -> Dict:
        res = self.model.transcribe(audio_path, language="en", word_timestamps=False)
        print(f"ASR segments: {len(res.get('segments',[]))}")
        return res

# ---------------- subtitles (optional, оставил твою реализацию) ----------------

class SubtitleGenerator:
    @staticmethod
    def _ts(s: float)->str:
        h=int(s//3600); m=int((s%3600)//60); sec=int(s%60); ms=int(round((s-int(s))*1000))
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    def write_srt(self, segments: List[Dict], out_path: str) -> str:
        ensure_dir(os.path.dirname(out_path))
        with open(out_path,"w",encoding="utf-8") as f:
            for i, seg in enumerate(segments,1):
                f.write(
                    f"{i}\n{self._ts(seg['start'])} --> {self._ts(seg['end'])}\n{seg['text'].strip().replace(' ',' ')}\n\n"
                )
        print(f"SRT: {out_path}")
        return out_path

class VideoSubtitler:
    @staticmethod
    def _esc(p:str)->str: return p.replace("\\","\\\\").replace(":","\\:")

    def burn(self, in_video: str, srt_path: str, out_video: str)->str:
        ensure_dir(os.path.dirname(out_video))
        vf = f"subtitles='{self._esc(srt_path)}':force_style='Fontsize=28,Outline=2,Shadow=1,MarginV=25'"
        run(["ffmpeg","-y","-i",in_video,"-vf",vf,"-c:a","copy",out_video])
        return out_video

# ---------------- Text→Pose (MS2SL external adapter) ----------------

@dataclass
class MS2SLConfig:
    repo_dir: str             # путь к клону MS2SL
    script: str               # полный путь к их скрипту инференса (например, text_infer.py)
    cmd_template: str         # шаблон запуска
    out_ext: str = "json"     # временный формат поз, до конвертации в .pose
    fps: int = 25

class MS2SLAdapter:
    """
    Вызывает внешний скрипт MS2SL, получаем keypoints (например, JSON),
    а затем конвертируем в .pose (формат pose-to-video).
    """

    def __init__(self, cfg: MS2SLConfig):
        self.cfg = cfg
        if not os.path.isdir(cfg.repo_dir): raise FileNotFoundError(cfg.repo_dir)
        if not os.path.isfile(cfg.script):  raise FileNotFoundError(cfg.script)

    def infer_to_json(self, text: str, out_json: str) -> str:
        ensure_dir(os.path.dirname(out_json))
        safe = text.replace('"','\\"')
        cmd = self.cfg.cmd_template.format(
            script=self.cfg.script, text=safe, fps=self.cfg.fps, out=out_json
        )
        print(f"[MS2SL] {cmd}")
        if os.name=="nt": run(["cmd","/c",f"cd /d \"{self.cfg.repo_dir}\" && {cmd}"])
        else:              run(["bash","-lc",f"cd \"{self.cfg.repo_dir}\" && {cmd}"])
        if not os.path.isfile(out_json):
            raise RuntimeError(f"MS2SL output not found: {out_json}")
        return out_json

    def json_to_pose(self, in_json: str, out_pose: str) -> str:
        """
        Конвертирует JSON ключевых точек MS2SL к простому .pose формату:
        .pose = JSON: { "fps": int, "frames": [ { "pose": [[x,y], ...] }, ... ] }
        Прим.: Точный маппинг скелета зависит от того, какие точки выдаёт MS2SL.
        Здесь показан шаблон-конвертер; при необходимости подстрой порядок/кол-во точек.
        """
        ensure_dir(os.path.dirname(out_pose))
        with open(in_json,"r",encoding="utf-8") as f:
            data = json.load(f)

        # ожидаем data["frames"] с keypoints, например: [{"hands":[...], "body":[...], "face":[...]}, ...]
        frames = []
        for fr in data.get("frames", []):
            # объединяем видимые 2D-точки в единый массив (упрощённо)
            pts = []
            for part in ("body","hands","face"):
                if part in fr and isinstance(fr[part], list):
                    for p in fr[part]:
                        if p and len(p)>=2:
                            pts.append([float(p[0]), float(p[1])])
            frames.append({"pose": pts})

        out = {"fps": self.cfg.fps, "frames": frames}
        with open(out_pose,"w",encoding="utf-8") as f:
            json.dump(out,f)
        return out_pose

    def text_to_pose(self, text: str, out_pose: str) -> str:
        tmp_json = out_pose.replace(".pose",".json") if out_pose.endswith(".pose") else out_pose+".json"
        self.infer_to_json(text, tmp_json)
        return self.json_to_pose(tmp_json, out_pose)

# ---------------- Pose→Video (pose-to-video CLI) ----------------

@dataclass
class Pose2VideoConfig:
    repo_dir: str       # путь к репо pose-to-video
    model_path: str     # путь к pix_to_pix.h5 (или к controlnet модели)
    impl_type: str = "pix2pix"  # "pix2pix" или "controlnet"

class Pose2Video:
    def __init__(self, cfg: Pose2VideoConfig):
        self.cfg = cfg
        if not os.path.isdir(cfg.repo_dir): raise FileNotFoundError(cfg.repo_dir)
        if not os.path.isfile(cfg.model_path): raise FileNotFoundError(cfg.model_path)

    def render(self, pose_file: str, out_video: str) -> str:
        ensure_dir(os.path.dirname(out_video))
        cmd = (
            f'pose_to_video --type={self.cfg.impl_type} --model="{self.cfg.model_path}" '
            f'--pose="{pose_file}" --video="{out_video}"'
        )
        print(f"[pose-to-video] {cmd}")
        if os.name=="nt": run(["cmd","/c",f"cd /d \"{self.cfg.repo_dir}\" && {cmd}"])
        else:              run(["bash","-lc",f"cd \"{self.cfg.repo_dir}\" && {cmd}"])
        if not os.path.isfile(out_video):
            raise RuntimeError(f"pose-to-video output not found: {out_video}")
        return out_video

# ---------------- video ops & overlay ----------------

class VideoOps:
    @staticmethod
    def match_duration(src: str, target_sec: float, out_path: str) -> str:
        ensure_dir(os.path.dirname(out_path))
        dur = ffprobe_duration(src)
        if dur<=0:
            shutil.copy(src,out_path); return out_path
        k = max(1e-6, target_sec/dur)  # setpts multiplier
        run(["ffmpeg","-y","-i",src,"-filter:v",f"setpts={k}*PTS,fps=25","-an",out_path])
        return out_path

    @staticmethod
    def concat(files: List[str], out_path: str) -> str:
        ensure_dir(os.path.dirname(out_path))
        lst = os.path.join(os.path.dirname(out_path) or ".", "_concat.txt")
        with open(lst,"w",encoding="utf-8") as f:
            for p in files: f.write(f"file '{os.path.abspath(p)}'\n")
        run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy",out_path])
        os.remove(lst)
        return out_path

class OverlayPiP:
    def apply(self, base_video: str, sign_video: str, out_video: str,
              width: int = 420, margin: int = 20, use_chromakey: bool=False) -> str:
        ensure_dir(os.path.dirname(out_video))
        scale = f"scale={width}:-1"
        if use_chromakey:
            fc = f"[1:v]{scale},chromakey=0x00FF00:0.3:0.1[sg];[0:v][sg]overlay=W-w-{margin}:H-h-{margin}"
        else:
            fc = f"[1:v]{scale}[sg];[0:v][sg]overlay=W-w-{margin}:H-h-{margin}"
        run(["ffmpeg","-y","-i",base_video,"-i",sign_video,"-filter_complex",fc,"-c:a","copy",out_video])
        return out_video

# ---------------- main pipeline ----------------

class EnglishToASLPipeline:
    def __init__(self, whisper_model="medium",
                 ms2sl_cfg: MS2SLConfig=None,
                 p2v_cfg: Pose2VideoConfig=None):
        self.asr = SpeechRecognizer(whisper_model)
        self.audio = AudioExtractor()
        self.ms2sl = MS2SLAdapter(ms2sl_cfg)
        self.p2v = Pose2Video(p2v_cfg)
        self.ops = VideoOps()
        self.overlay = OverlayPiP()
        self.subs = SubtitleGenerator()
        self.burner = VideoSubtitler()

    def process(self, input_video: str, output_video: str,
                tmp_dir: str = "_work_asl",
                overlay_width: int = 420,
                also_burn_subtitles: bool = False,
                srt_segments: Optional[List[Dict]] = None,
                use_chromakey: bool = False) -> str:

        ensure_dir(tmp_dir)

        # 1) audio
        wav = os.path.join(tmp_dir,"audio.wav")
        self.audio.extract(input_video, wav)

        # 2) ASR
        asr = self.asr.transcribe(wav)
        segments = asr.get("segments", [])
        if not segments: raise RuntimeError("No speech segments recognized")

        # 3) (optional) burn subtitles
        base_for_overlay = input_video
        if also_burn_subtitles:
            if srt_segments is None:
                srt_segments = [{"start":s["start"],"end":s["end"],"text":s["text"]} for s in segments]
            srt_path = os.path.join(tmp_dir,"subs.srt")
            self.subs.write_srt(srt_segments, srt_path)
            base_for_overlay = os.path.join(tmp_dir,"with_subs.mp4")
            self.burner.burn(input_video, srt_path, base_for_overlay)

        # 4) per-segment: text -> pose (.pose) -> video
        sign_clips=[]
        for i, seg in enumerate(segments):
            text = seg.get("text","").strip()
            if not text: continue

            pose_path = os.path.join(tmp_dir,f"seg_{i:04d}.pose")
            self.ms2sl.text_to_pose(text, pose_path)

            raw_vid = os.path.join(tmp_dir,f"seg_{i:04d}_raw.mp4")
            self.p2v.render(pose_path, raw_vid)

            target = max(0.2, float(seg["end"] - seg["start"]))
            conf_vid = os.path.join(tmp_dir,f"seg_{i:04d}_conf.mp4")
            self.ops.match_duration(raw_vid, target, conf_vid)
            sign_clips.append(conf_vid)

        if not sign_clips: raise RuntimeError("No sign clips generated")

        # 5) concatenate sign track
        sign_full = os.path.join(tmp_dir,"sign_full.mp4")
        self.ops.concat(sign_clips, sign_full)

        # 6) overlay
        final_path = self.overlay.apply(
            base_for_overlay, sign_full, output_video,
            width=overlay_width, use_chromakey=use_chromakey
        )

        print(f"FINAL: {final_path}")
        return final_path


# ---------------- usage ----------------
if __name__ == "__main__":
    INPUT_VIDEO  = r"D:\data\input_english.mp4"
    OUTPUT_VIDEO = r"D:\data\output_with_asl.mp4"

    # === MS2SL (Text→Pose) ===
    # ВАЖНО: подставь название их инференс-скрипта и флаги из README.
    # Ниже пример шаблона команды:
    #   python text_infer.py --text "..." --fps 25 --out "poses.json"
    MS2SL_REPO = r"D:\models\MS2SL"
    MS2SL_SCRIPT = os.path.join(MS2SL_REPO, "text_infer.py")  # поменяй на реальное имя
    CMD_TEMPLATE_MS2SL = 'python "{script}" --text "{text}" --fps {fps} --out "{out}"'

    ms2sl_cfg = MS2SLConfig(
        repo_dir=MS2SL_REPO,
        script=MS2SL_SCRIPT,
        cmd_template=CMD_TEMPLATE_MS2SL,
        out_ext="json",
        fps=25
    )

    # === pose-to-video (Pose→Video) ===
    P2V_REPO = r"D:\models\pose-to-video"
    P2V_MODEL = os.path.join(P2V_REPO, "pix_to_pix.h5")
    p2v_cfg = Pose2VideoConfig(
        repo_dir=P2V_REPO,
        model_path=P2V_MODEL,
        impl_type="pix2pix"
    )

    pipeline = EnglishToASLPipeline(
        whisper_model="medium",
        ms2sl_cfg=ms2sl_cfg,
        p2v_cfg=p2v_cfg
    )

    pipeline.process(
        input_video=INPUT_VIDEO,
        output_video=OUTPUT_VIDEO,
        tmp_dir="_work_asl",
        overlay_width=420,
        also_burn_subtitles=False,
        srt_segments=None,
        use_chromakey=False
    )
