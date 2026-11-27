import queue
import threading
import time


class SpeechToTextWorker(threading.Thread):
    """Offline microphone capture and Whisper transcription worker."""

    def __init__(self, output_queue):
        super().__init__(daemon=True, name="speech-to-text")
        self.output_queue = output_queue
        self.language = "en"
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=256)
        self.stream = None

    def stop(self):
        self.stop_event.set()

    def emit(self, *event):
        self.output_queue.put(event)

    @staticmethod
    def resample_audio(audio, source_rate):
        import numpy as np

        target_rate = 16000
        if source_rate == target_rate:
            return audio.astype(np.float32, copy=False)

        target_length = max(1, round(len(audio) * target_rate / source_rate))
        old_positions = np.arange(len(audio), dtype=np.float32)
        new_positions = np.linspace(
            0, len(audio) - 1, target_length, dtype=np.float32
        )
        return np.interp(new_positions, old_positions, audio).astype(np.float32)

    def run(self):
        try:
            import numpy as np
            import sounddevice as sd
            from faster_whisper import WhisperModel

            self.emit("speech_status", "Loading offline Whisper Base model...")
            model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )

            device = sd.query_devices(kind="input")
            sample_rate = int(device["default_samplerate"])
            if int(device["max_input_channels"]) < 1:
                raise RuntimeError("No microphone input device is available.")

            def audio_callback(indata, frames, time_info, status):
                if self.stop_event.is_set():
                    return
                try:
                    self.audio_queue.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    pass

            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=max(1024, int(sample_rate * 0.25)),
                callback=audio_callback,
            )
            self.stream.start()
            self.emit("speech_status", "Listening — speak clearly into your microphone.")

            audio_parts = []
            sample_count = 0
            chunk_size = sample_rate * 6

            while not self.stop_event.is_set():
                try:
                    block = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                audio_parts.append(block)
                sample_count += len(block)

                if sample_count >= chunk_size:
                    audio = np.concatenate(audio_parts)
                    audio_parts = []
                    sample_count = 0
                    self.transcribe_chunk(model, audio, sample_rate)

            if sample_count >= int(sample_rate * 1.5):
                audio = np.concatenate(audio_parts)
                self.transcribe_chunk(model, audio, sample_rate)

        except Exception as exc:
            self.emit("speech_error", f"{type(exc).__name__}: {exc}")

        finally:
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            self.emit("speech_stopped")

    def transcribe_chunk(self, model, audio, sample_rate):
        if self.stop_event.is_set():
            return

        import numpy as np

        audio = self.resample_audio(audio, sample_rate)
        if audio.size == 0 or np.max(np.abs(audio)) < 0.008:
            return

        segments, info = model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=3,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        if text:
            self.emit("speech_result", text, info.language or "unknown")
