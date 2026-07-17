import faster_whisper
import sounddevice as sd
import wavio
import numpy as np


class AudioHandler:
    def __init__(self):
        self.model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")

    def capture_audio(self):
        try:
            filename = self._record_audio()
        except Exception as e:
            print(f"Error recording audio: {e}")
            raise e

        transcript = self._transcribe_audio(filename)
        return transcript

    def _transcribe_audio(self, filename: str):
        segments, _ = self.model.transcribe(filename)
        return " ".join([s.text for s in segments])

    def _record_audio(
        self,
        filename: str = "user_audio.wav",
        fs: int = 16000,
        max_seconds: int = 30,
        silence_seconds: float = 1.0,
        threshold: float = 500,  # int16 RMS; tune this
    ):
        """Record user audio with silence detection done by measuring volume of each chunk of the streamed audio."""
        chunk = int(0.1 * fs)  # 100ms frames
        silence_chunks_needed = int(silence_seconds / 0.1)
        max_chunks = int(max_seconds / 0.1)

        # Save each chunk of audio to a frames list
        frames = []
        silent = 0
        started = False

        # Start audio stream
        with sd.InputStream(samplerate=fs, channels=1, dtype="int16", blocksize=chunk) as stream:
            # Read audio in chunks
            for _ in range(max_chunks):
                data, _ = stream.read(chunk)
                frames.append(data.copy())
                volume = np.sqrt(np.mean(data.astype(np.float64) ** 2))

                # If volume is above threshold, start recording
                if volume > threshold:
                    started = True
                    silent = 0
                # If volume is below threshold and recording has started, silence counter is incremented
                elif started:
                    silent += 1
                    # If silence counter is greater than or equal to the number of chunks needed to detect silence, break the loop
                    if silent >= silence_chunks_needed:
                        break

        if not frames:
            print("No audio captured")
            return

        # Concatenate all frames into a single numpy array
        recording = np.concatenate(frames, axis=0)
        # Save the recording to a WAV file
        wavio.write(filename, recording, fs, sampwidth=2)
        print(f"Recording saved to {filename}")
        return filename