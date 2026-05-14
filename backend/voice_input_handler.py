# voice_input_handler.py — FIXED v6
#
# BUGS FIXED vs v5:
# ─────────────────────────────────────────────────────────────────────────────
# BUG 1 FIX — ffmpeg volume=100 → volume=3.0
#   volume=100 means 100× amplification (+40 dB). Every real mic recording
#   becomes a hard-clipped square wave pegged at ±32767. librosa then receives
#   distorted audio → piptrack returns pitch=0 → biomarker emotion="neutral"
#   → EmotionAnalyzer returns "neutral" every time.
#   volume=3.0 = 3× gain (+9.5 dB), enough to lift quiet laptop-mic audio
#   without clipping normal speech.
#
# BUG 2 FIX — silence guard in run_pipeline() now also requires empty transcript
#   Previous: is_truly_silent = pitch==0 AND tone < threshold
#   Fixed:    is_truly_silent = pitch==0 AND tone < threshold AND transcript==""
#   Prevents falsely short-circuiting to "neutral" when STT did pick up words
#   even if acoustic features are weak.
#
# ALL OTHER CODE IDENTICAL TO v5.

import os
import io
import subprocess
import tempfile
import wave
import struct

import numpy as np

from backend.stt              import STT
from backend.voice_biomarker  import VoiceBiomarker
from backend.emotion_analyzer import EmotionAnalyzer


class VoiceInputHandler:

    TARGET_SR     = 16_000
    MIN_WAV_BYTES = 3_200

    INT16_RMS_THRESHOLD   = 0.3
    FLOAT32_RMS_THRESHOLD = 0.000001

    def __init__(self):
        self.raw_audio   = None
        self.sample_rate = self.TARGET_SR

    # ── Strategy 1: ffmpeg ───────────────────────────────────────────────────
    @staticmethod
    def _decode_ffmpeg_pipe(input_path: str, output_wav_path: str,
                            target_sr: int = 16_000) -> tuple:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "volume=3.0",   # BUG 1 FIX: was volume=100 (100× = hard clip)
            "-ar", str(target_sr),
            "-ac", "1",
            "-f",  "wav",
            output_wav_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except FileNotFoundError:
            print("[ffmpeg] ffmpeg not found in PATH.", flush=True)
            return False, 0.0
        except subprocess.TimeoutExpired:
            print("[ffmpeg] ffmpeg timed out.", flush=True)
            return False, 0.0

        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            print(f"[ffmpeg] returned {result.returncode}: {err[:300]}", flush=True)
            return False, 0.0

        if not os.path.exists(output_wav_path):
            print("[ffmpeg] output WAV not created.", flush=True)
            return False, 0.0

        file_size = os.path.getsize(output_wav_path)
        if file_size < VoiceInputHandler.MIN_WAV_BYTES:
            print(f"[ffmpeg] output WAV too small: {file_size} bytes", flush=True)
            return False, 0.0

        try:
            with wave.open(output_wav_path, "rb") as wf:
                n_frames = wf.getnframes()
                raw      = wf.readframes(n_frames)
                fr       = wf.getframerate()

            audio = np.frombuffer(raw, dtype=np.int16)
            rms   = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            dur   = n_frames / fr

            print(
                f"[ffmpeg] WAV OK: {file_size} bytes  "
                f"frames={n_frames}  duration={dur:.2f}s  rms={rms:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[ffmpeg] WAV readback failed: {e}", flush=True)
            return False, 0.0

        if rms < VoiceInputHandler.INT16_RMS_THRESHOLD:
            print(
                f"[ffmpeg] rms={rms:.4f} < threshold={VoiceInputHandler.INT16_RMS_THRESHOLD}"
                f" → SILENT ✗",
                flush=True,
            )
            return False, rms

        print(f"[ffmpeg] ✓  rms={rms:.4f}", flush=True)
        return True, rms

    # ── Strategy 2: PyAV float32 ─────────────────────────────────────────────
    @staticmethod
    def _decode_pyav_float32(input_path: str, output_wav_path: str,
                             target_sr: int = 16_000) -> tuple:
        try:
            import av
        except ImportError:
            print("[PyAV-f32] av not installed.", flush=True)
            return False, 0.0

        all_frames = []
        orig_sr    = None

        try:
            container     = av.open(input_path)
            audio_streams = [s for s in container.streams if s.type == "audio"]

            if not audio_streams:
                container.close()
                return False, 0.0

            stream  = audio_streams[0]
            orig_sr = stream.codec_context.sample_rate
            print(
                f"[PyAV-f32] codec={stream.codec_context.name}  "
                f"sr={orig_sr}  ch={stream.codec_context.channels}",
                flush=True,
            )

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=orig_sr)

            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    if arr.ndim == 2: arr = arr[0]
                    all_frames.append(arr.astype(np.float32))

            for rf in resampler.resample(None):
                arr = rf.to_ndarray()
                if arr.ndim == 2: arr = arr[0]
                all_frames.append(arr.astype(np.float32))

            container.close()
        except Exception as exc:
            import traceback
            print(f"[PyAV-f32] decode error: {exc}", flush=True)
            traceback.print_exc()
            return False, 0.0

        if not all_frames:
            print("[PyAV-f32] Zero frames decoded.", flush=True)
            return False, 0.0

        audio_f32 = np.concatenate(all_frames)
        rms_f32   = float(np.sqrt(np.mean(audio_f32 ** 2)))
        print(f"[PyAV-f32] {len(audio_f32)} float32 samples  rms={rms_f32:.6f}", flush=True)

        if rms_f32 < VoiceInputHandler.FLOAT32_RMS_THRESHOLD:
            print(f"[PyAV-f32] SILENT ✗  rms={rms_f32:.6f}", flush=True)
            return False, rms_f32

        if orig_sr and orig_sr != target_sr:
            try:
                import librosa
                audio_f32 = librosa.resample(audio_f32, orig_sr=orig_sr, target_sr=target_sr)
            except Exception as e:
                print(f"[PyAV-f32] Resample failed ({e})", flush=True)

        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        rms_i16   = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
        print(f"[PyAV-f32] int16 rms={rms_i16:.2f}", flush=True)

        try:
            with wave.open(output_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_sr)
                wf.writeframes(audio_i16.tobytes())
            print(f"[PyAV-f32] WAV written: {os.path.getsize(output_wav_path)} bytes ✓", flush=True)
            return True, rms_i16
        except Exception as e:
            print(f"[PyAV-f32] WAV write failed: {e}", flush=True)
            return False, 0.0

    # ── Strategy 3: soundfile direct ─────────────────────────────────────────
    @staticmethod
    def _decode_soundfile_direct(input_path: str, output_wav_path: str,
                                 target_sr: int = 16_000) -> tuple:
        try:
            import soundfile as sf
            data, file_sr = sf.read(input_path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"[soundfile-direct] Cannot read {input_path}: {e}", flush=True)
            return False, 0.0

        if data.ndim > 1:
            data = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(data ** 2)))
        print(f"[soundfile-direct] {len(data)} samples  sr={file_sr}  rms={rms:.6f}", flush=True)

        if rms < VoiceInputHandler.FLOAT32_RMS_THRESHOLD:
            print("[soundfile-direct] SILENT ✗", flush=True)
            return False, rms

        if file_sr != target_sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=file_sr, target_sr=target_sr)
            except Exception as e:
                print(f"[soundfile-direct] resample failed: {e}", flush=True)

        audio_i16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        rms_i16   = float(np.sqrt(np.mean(audio_i16.astype(np.float32) ** 2)))
        try:
            with wave.open(output_wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_sr)
                wf.writeframes(audio_i16.tobytes())
            print(f"[soundfile-direct] WAV written ✓  rms={rms_i16:.2f}", flush=True)
            return True, rms_i16
        except Exception as e:
            print(f"[soundfile-direct] WAV write failed: {e}", flush=True)
            return False, 0.0

    # ── Strategy 4: WAV direct ───────────────────────────────────────────────
    @staticmethod
    def _try_read_as_wav(input_path: str, output_wav_path: str,
                         target_sr: int = 16_000) -> tuple:
        try:
            with wave.open(input_path, "rb") as wf:
                ch  = wf.getnchannels()
                sw  = wf.getsampwidth()
                fr  = wf.getframerate()
                nf  = wf.getnframes()
                raw = wf.readframes(nf)
        except Exception:
            return False, 0.0

        if sw != 2:
            return False, 0.0

        audio = np.frombuffer(raw, dtype=np.int16)
        if ch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        print(f"[wav-direct] sr={fr}  samples={len(audio)}  rms={rms:.2f}", flush=True)

        if rms < VoiceInputHandler.INT16_RMS_THRESHOLD:
            return False, rms

        if fr != target_sr:
            try:
                import librosa
                af32  = audio.astype(np.float32) / 32768.0
                af32  = librosa.resample(af32, orig_sr=fr, target_sr=target_sr)
                audio = (af32 * 32767).clip(-32768, 32767).astype(np.int16)
            except Exception as e:
                print(f"[wav-direct] resample failed: {e}", flush=True)

        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(audio.tobytes())

        print("[wav-direct] WAV written ✓", flush=True)
        return True, rms

    # ── Strategy 5: pydub ────────────────────────────────────────────────────
    def _decode_with_pydub(self, input_path: str, out_path: str) -> tuple:
        try:
            from pydub import AudioSegment
        except ImportError:
            print("[pydub] Not installed — pip install pydub", flush=True)
            return False, 0.0

        try:
            audio = AudioSegment.from_file(input_path)
            audio = (audio.set_frame_rate(16000).set_channels(1).set_sample_width(2))
            audio.export(out_path, format="wav")
            size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if size < self.MIN_WAV_BYTES:
                return False, 0.0
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            rms     = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
            print(f"[pydub] samples={len(samples)}  rms={rms:.2f}", flush=True)
            if rms < self.INT16_RMS_THRESHOLD:
                print(f"[pydub] SILENT ✗", flush=True)
                return False, rms
            print(f"[pydub] ✓  rms={rms:.2f}", flush=True)
            return True, rms
        except Exception as exc:
            print(f"[pydub] Failed: {exc}", flush=True)
            return False, 0.0

    # ── Verify WAV ───────────────────────────────────────────────────────────
    @staticmethod
    def _verify_wav(wav_path: str) -> tuple:
        try:
            with wave.open(wav_path, "rb") as wf:
                ch  = wf.getnchannels()
                sw  = wf.getsampwidth()
                fr  = wf.getframerate()
                nf  = wf.getnframes()
                raw = wf.readframes(nf)

            if sw != 2:
                return False, 0.0

            audio = np.frombuffer(raw, dtype=np.int16)
            if ch == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

            rms      = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            duration = nf / fr

            print(
                f"[verify-wav] frames={nf}  sr={fr}  ch={ch}  "
                f"duration={duration:.2f}s  rms={rms:.4f}",
                flush=True,
            )

            has_audio = rms > VoiceInputHandler.INT16_RMS_THRESHOLD
            print(f"[verify-wav] → {'AUDIO CONFIRMED ✓' if has_audio else 'STILL SILENT ✗'}", flush=True)
            return has_audio, rms

        except Exception as e:
            print(f"[verify-wav] read failed: {e}", flush=True)
            return False, 0.0

    # ── preprocess_audio ─────────────────────────────────────────────────────
    def preprocess_audio(self, input_path: str) -> str:
        tmp      = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = tmp.name
        tmp.close()

        file_size = os.path.getsize(input_path)
        print(f"[VoiceInputHandler] preprocess: {input_path} ({file_size} bytes)", flush=True)

        print("[VoiceInputHandler] Strategy 1: ffmpeg → WAV file...", flush=True)
        ok, rms = self._decode_ffmpeg_pipe(input_path, out_path)
        if ok:
            has_audio, verified_rms = self._verify_wav(out_path)
            if has_audio:
                print(f"[VoiceInputHandler] ✓ Strategy 1 succeeded  rms={verified_rms:.2f}", flush=True)
                return out_path

        print("[VoiceInputHandler] Strategy 2: PyAV float32...", flush=True)
        ok, rms = self._decode_pyav_float32(input_path, out_path)
        if ok:
            has_audio, verified_rms = self._verify_wav(out_path)
            if has_audio:
                print(f"[VoiceInputHandler] ✓ Strategy 2 succeeded  rms={verified_rms:.2f}", flush=True)
                return out_path

        print("[VoiceInputHandler] Strategy 3: soundfile direct...", flush=True)
        ok, rms = self._decode_soundfile_direct(input_path, out_path)
        if ok:
            has_audio, verified_rms = self._verify_wav(out_path)
            if has_audio:
                print(f"[VoiceInputHandler] ✓ Strategy 3 succeeded  rms={verified_rms:.2f}", flush=True)
                return out_path

        print("[VoiceInputHandler] Strategy 4: WAV direct...", flush=True)
        ok, rms = self._try_read_as_wav(input_path, out_path)
        if ok:
            has_audio, verified_rms = self._verify_wav(out_path)
            if has_audio:
                print(f"[VoiceInputHandler] ✓ Strategy 4 succeeded  rms={verified_rms:.2f}", flush=True)
                return out_path

        print("[VoiceInputHandler] Strategy 5: pydub...", flush=True)
        ok, rms = self._decode_with_pydub(input_path, out_path)
        if ok:
            has_audio, verified_rms = self._verify_wav(out_path)
            if has_audio:
                print(f"[VoiceInputHandler] ✓ Strategy 5 succeeded  rms={verified_rms:.2f}", flush=True)
                return out_path

        print("[VoiceInputHandler] ALL strategies failed. Returning last output path.", flush=True)
        return out_path

    # ── run_pipeline ─────────────────────────────────────────────────────────
    def run_pipeline(self, audio_path: str, lang: str = "en") -> dict:
        cleaned_path = None
        try:
            cleaned_path = self.preprocess_audio(audio_path)

            stt        = STT()
            stt_result = stt.convert_to_text(cleaned_path, language=lang)

            if "error" in stt_result:
                raise RuntimeError(f"STT failed: {stt_result['error']}")

            transcript = stt_result.get("transcript", "")
            print(f"[VoiceInputHandler] transcript='{transcript[:80]}'  lang={lang}", flush=True)

            biomarker  = VoiceBiomarker()
            biomarker.extract_mfcc(cleaned_path)
            bio_result = biomarker.analyze_voice_emotion()

            print(
                f"[VoiceInputHandler] biomarker: pitch={bio_result['pitch']:.1f} Hz  "
                f"tone={bio_result['tone']:.5f}  emotion={bio_result['emotion_from_voice']}",
                flush=True,
            )

            # BUG 2 FIX: require BOTH silent acoustics AND empty transcript
            # before short-circuiting to "neutral".
            # Previously: pitch==0 AND tone < threshold  (ignored transcript)
            # Now:        pitch==0 AND tone < threshold AND transcript==""
            is_truly_silent = (
                bio_result["pitch"] == 0.0
                and bio_result["tone"] < VoiceBiomarker.SILENCE_TONE_THRESHOLD
                and not transcript.strip()
            )

            if is_truly_silent:
                print("[VoiceInputHandler] Confirmed silent → returning neutral.", flush=True)
                return {
                    "transcript":       "",
                    "dominant_emotion": "neutral",
                    "fusion":           {"anxiety": 0.0, "stress": 0.0, "sadness": 0.0},
                    "biomarkers": {
                        "pitch":     0.0,
                        "tone":      0.0,
                        "mfcc_mean": bio_result.get("mfcc_mean", 0.0),
                    },
                }

            analyzer   = EmotionAnalyzer()
            emo_result = analyzer.classify_emotion(transcript, bio_result, language=lang)

            return {
                "transcript":       transcript,
                "dominant_emotion": emo_result["final_emotion_label"],
                "fusion":           emo_result["fusion"],
                "biomarkers": {
                    "pitch":     bio_result["pitch"],
                    "tone":      bio_result["tone"],
                    "mfcc_mean": bio_result.get("mfcc_mean", 0.0),
                },
            }

        except Exception as exc:
            import traceback
            print(f"[VoiceInputHandler] PIPELINE ERROR: {exc}", flush=True)
            traceback.print_exc()
            return {
                "error":            str(exc),
                "transcript":       "",
                "dominant_emotion": "unknown",
                "fusion":           {"anxiety": 0.0, "stress": 0.0, "sadness": 0.0},
                "biomarkers":       {"pitch": 0.0, "tone": 0.0, "mfcc_mean": 0.0},
            }

        finally:
            if cleaned_path and os.path.exists(cleaned_path):
                os.remove(cleaned_path)
                print("[VoiceInputHandler] Temp WAV removed.", flush=True)