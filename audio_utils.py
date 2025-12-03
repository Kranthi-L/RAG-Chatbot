# audio_utils.py
# Phase 2: Speech-to-Text (STT) utilities
import os
import time
import numpy as np
import sounddevice as sd
import whisper
from typing import Optional, Tuple, List
import tempfile
import wave
try:
    import scipy.io.wavfile as wavfile
except ImportError:
    wavfile = None

# TTS imports (optional - for cross-platform support)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Load Whisper model (cache it to avoid reloading)
_whisper_model = None

def get_whisper_model():
    """Get or load the Whisper model (cached for performance)."""
    global _whisper_model
    if _whisper_model is None:
        # Use base model for balance between speed and accuracy
        # Options: tiny, base, small, medium, large
        model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds (default 5 seconds)
        sample_rate: Sample rate in Hz (default 16000, Whisper's preferred rate)
    
    Returns:
        Tuple of (audio_data, sample_rate)
    
    Raises:
        Exception: If microphone access fails
    """
    try:
        print(f"Recording for {duration} seconds... Speak now!")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        return audio_data.flatten(), sample_rate
    except Exception as e:
        raise Exception(f"Failed to record audio: {str(e)}. Please check microphone permissions.")

def transcribe_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
    
    Returns:
        Transcribed text string
    
    Raises:
        Exception: If transcription fails
    """
    try:
        # Validate audio data
        if len(audio_data) == 0:
            raise Exception("Empty audio data provided")
        
        # Check audio statistics for debugging
        max_val = np.abs(audio_data).max()
        mean_val = np.abs(audio_data).mean()
        rms = np.sqrt(np.mean(audio_data**2))
        
        print(f"Audio stats - Duration: {len(audio_data)/sample_rate:.2f}s, Max: {max_val:.4f}, Mean: {mean_val:.4f}, RMS: {rms:.4f}")
        
        # Check if audio is all zeros or essentially silent
        if max_val < 1e-6:
            raise Exception(f"Audio appears to be empty or not recorded (max value: {max_val:.8f}). Check microphone connection and permissions.")
        
        # If audio is very quiet, amplify it significantly
        if max_val < 0.05:
            print(f"⚠️ Audio is very quiet (max: {max_val:.4f}), amplifying...")
            # Amplify to bring max to around 0.5 (reasonable level)
            target_level = 0.5
            amplification_factor = target_level / max_val if max_val > 0 else 1.0
            # Cap at 20x to avoid excessive noise amplification
            amplification_factor = min(amplification_factor, 20.0)
            audio_data = audio_data * amplification_factor
            max_val = np.abs(audio_data).max()
            print(f"After amplification - Max: {max_val:.4f} (amplified {amplification_factor:.1f}x)")
        
        # Normalize audio to prevent clipping (but preserve relative levels)
        if max_val > 1.0:
            # Audio is clipping, normalize down
            audio_data = audio_data / max_val * 0.95
            print("Normalized audio to prevent clipping")
        elif max_val < 0.1:
            # Audio is too quiet, normalize up to reasonable level
            audio_data = audio_data / max_val * 0.5 if max_val > 0 else audio_data
            print("Amplified quiet audio")
        
        # Ensure audio is float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Ensure audio is in range [-1, 1]
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        model = get_whisper_model()
        
        # Whisper expects audio in the range [-1, 1] and at 16kHz
        # Try transcribing directly with numpy array first
        print(f"Transcribing {len(audio_data)/sample_rate:.2f} seconds of audio...")
        
        # Always use file-based transcription for better quality (Whisper works better with files)
        # This also allows us to save debug files if needed
        if wavfile is None:
            raise Exception("scipy is required for audio transcription. Please install: pip install scipy")
        
        # Save to temp file for transcription
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Scale to int16 range for WAV file, ensuring we don't clip
                max_abs = np.abs(audio_data).max()
                if max_abs > 0:
                    # Normalize to prevent clipping, but keep it loud enough
                    if max_abs < 1.0:
                        # Amplify if needed
                        audio_normalized = audio_data / max_abs * 0.9  # Use 90% to avoid clipping
                    else:
                        audio_normalized = audio_data / max_abs * 0.9
                    audio_int16 = (audio_normalized * 32767).astype(np.int16)
                else:
                    raise Exception("Audio data is all zeros - microphone may not be recording")
                
                wavfile.write(tmp_file.name, sample_rate, audio_int16)
                
                # Optionally save for debugging (set SAVE_AUDIO_DEBUG=1 in .env)
                if os.getenv("SAVE_AUDIO_DEBUG", "0") == "1":
                    debug_file = f"debug_audio_{int(time.time())}.wav"
                    import shutil
                    shutil.copy(tmp_file.name, debug_file)
                    print(f"💾 Saved debug audio to: {debug_file}")
                
                # Transcribe from file (more reliable than numpy array)
                # Note: Removed initial_prompt as it was causing hallucinations when stopping listening
                result = model.transcribe(
                    tmp_file.name, 
                    language="en", 
                    verbose=False, 
                    fp16=False,
                    temperature=0.0  # Use deterministic decoding for consistency
                )
                
                os.unlink(tmp_file.name)  # Clean up temp file
        except Exception as e:
            error_msg = str(e)
            # Check if it's an ffmpeg error
            if "ffmpeg" in error_msg.lower() or "no such file or directory" in error_msg.lower():
                raise Exception(
                    f"FFmpeg is required for audio transcription but is not installed.\n\n"
                    f"To install FFmpeg:\n"
                    f"  macOS: brew install ffmpeg\n"
                    f"  Linux: sudo apt-get install ffmpeg (or use your package manager)\n"
                    f"  Windows: Download from https://ffmpeg.org/download.html\n\n"
                    f"After installing, restart the application.\n\n"
                    f"Original error: {error_msg}"
                )
            else:
                raise Exception(f"Failed to transcribe audio file: {error_msg}")
        
        # Extract transcription - check both main text and segments
        transcribed_text = result.get("text", "").strip()
        
        # Debug: print what Whisper detected
        if "segments" in result:
            segments = result["segments"]
            print(f"Whisper detected {len(segments)} segment(s)")
            for i, seg in enumerate(segments[:3]):  # Show first 3 segments
                print(f"  Segment {i+1}: '{seg.get('text', '').strip()}' (confidence: {seg.get('no_speech_prob', 'N/A')})")
        
        # If main text is empty or seems wrong, try segments
        if not transcribed_text or len(transcribed_text) < 2:
            if "segments" in result:
                segments = result["segments"]
                if len(segments) > 0:
                    segment_texts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
                    if segment_texts:
                        transcribed_text = " ".join(segment_texts)
                        print(f"Using text from segments: {transcribed_text}")
        
        if not transcribed_text:
            # Provide detailed error info
            error_details = f"Audio stats: max={max_val:.4f}, mean={mean_val:.4f}, rms={rms:.4f}, duration={len(audio_data)/sample_rate:.2f}s"
            if max_val < 0.001:
                raise Exception(f"Audio too quiet for transcription. {error_details}. Try speaking louder or closer to microphone.")
            elif len(audio_data) < sample_rate * 0.5:
                raise Exception(f"Audio too short for transcription. {error_details}. Please speak for at least 1 second.")
            else:
                raise Exception(f"Transcription returned empty. {error_details}. Audio may be unclear or contain only noise.")
        
        # Validate transcription - filter out common Whisper hallucinations
        if not is_valid_transcription(transcribed_text):
            # Check if this is likely a hallucination
            raise Exception(
                f"Transcription appears to be a hallucination or background noise: '{transcribed_text}'. "
                f"Please speak clearly or check your microphone."
            )
        
        print(f"✅ Transcription: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        raise Exception(f"Failed to transcribe audio: {str(e)}")


def is_valid_transcription(text: str) -> bool:
    """
    Validate if a transcription is likely real speech vs Whisper hallucination.
    
    Common Whisper hallucinations include phrases like:
    - "I'm not sure if i can get a computer"
    - "Thank you for watching"
    - Generic phrases that don't match actual speech
    
    Args:
        text: Transcribed text to validate
    
    Returns:
        True if transcription seems valid, False if likely hallucination
    """
    if not text or len(text.strip()) < 2:
        return False
    
    text_lower = text.lower().strip()
    
    # List of common Whisper hallucinations (add more as discovered)
    common_hallucinations = [
        "i'm not sure if i can get a computer",
        "i'm not sure if i can get",
        "thank you for watching",
        "thank you for",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "this is a question about computer networking",
        "this is a question about computer",
        "question about computer networking or computer architecture",
        "computer networking or computer architecture",
    ]
    
    # Check if transcription matches a known hallucination
    for hallucination in common_hallucinations:
        if hallucination in text_lower:
            print(f"⚠️ Detected potential Whisper hallucination: '{text}'")
            return False
    
    # Check if transcription is suspiciously generic or doesn't look like a question
    # Very short transcriptions might be noise
    if len(text.strip()) < 5:
        return False
    
    # If it passes all checks, consider it valid
    return True


def clean_text_for_speech(text: str) -> str:
    """
    Clean text for TTS by removing markdown, code blocks, LaTeX formulas, and special characters.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text suitable for speech synthesis
    """
    if not text:
        return ""
    
    import re

    # SPECIAL-CASE: transmission delay formula so we don't mangle the sentence
    # [ d_{trans} = \frac{L}{R} ] -> "d trans equals L divided by R"
    text = text.replace(
        "[ d_{trans} = \\frac{L}{R} ]",
        "d trans equals L divided by R"
    )
    
    # Remove LaTeX math blocks (\[ ... \] or \( ... \))
    text = re.sub(r'\\\[.*?\\\]', '[formula]', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', '[formula]', text, flags=re.DOTALL)
    
    # Remove LaTeX inline math ($ ... $ or $$ ... $$)
    text = re.sub(r'\$\$([^$]+)\$\$', '[formula]', text)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    
    # Handle LaTeX fractions: \frac{L}{R} -> L divided by R (for any remaining)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 divided by \2', text)
    
    # Handle LaTeX subscripts: d_{trans} -> d trans (for any remaining)
    text = re.sub(r'_\{([^}]+)\}', r' \1 ', text)
    text = re.sub(r'_([a-zA-Z0-9]+)', r' \1 ', text)
    
    # Remove LaTeX commands (like \text, \mathrm, etc.)
    text = re.sub(r'\\[a-zA-Z]+\{?[^}]*\}?', '', text)
    
    # Remove curly braces (used in LaTeX)
    text = re.sub(r'\{([^}]+)\}', r'\1', text)
    
    # Remove code blocks (```code```)
    text = re.sub(r'```[\s\S]*?```', '[code block]', text)
    
    # Remove inline code (`code`)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold (**text** or __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Remove markdown italic (*text* or _text_)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Note: _text_ italic is handled after LaTeX subscripts to avoid conflicts
    
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Convert common symbols to words
    text = text.replace('->', 'to')
    text = text.replace('<-', 'from')
    text = text.replace('==', 'equals')
    text = text.replace('!=', 'not equals')
    text = text.replace('<=', 'less than or equal to')
    text = text.replace('>=', 'greater than or equal to')
    text = text.replace('&', 'and')
    text = text.replace('|', 'or')
    text = text.replace('=', 'equals')
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def record_audio_with_silence_detection(
    sample_rate: int = 16000,
    chunk_duration: float = 0.1,  # Analyze audio in 100ms chunks
    silence_threshold: float = 0.01,  # Energy threshold for silence (higher = filters background noise better)
    silence_duration: float = 1.5,  # Stop after 1.5 seconds of silence AFTER speech
    max_duration: float = 60.0  # Maximum recording time (increased to 60s, only used as absolute safety limit)
) -> Tuple[np.ndarray, int]:
    """
    Record audio with automatic silence detection - stops when user stops talking.
    Uses continuous stream recording (like fallback) for better audio quality.
    Similar to Google Assistant behavior.
    
    Args:
        sample_rate: Sample rate in Hz (default 16000)
        chunk_duration: Duration of each audio chunk to analyze (seconds)
        silence_threshold: Energy threshold below which audio is considered silence
        silence_duration: How long silence must last before stopping (seconds)
        max_duration: Maximum recording time as safety limit (seconds)
    
    Returns:
        Tuple of (audio_data, sample_rate)
    
    Raises:
        Exception: If microphone access fails
    """
    try:
        chunk_samples = int(chunk_duration * sample_rate)
        silence_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        
        all_audio_chunks = []
        silent_chunk_count = 0
        has_speech = False
        max_energy_seen = 0.0
        
        print("🎤 Recording... Speak now! (Recording stops automatically when you finish speaking)")
        
        # Use continuous stream recording (keeps mic constantly active like fallback)
        # This prevents blinking mic indicator and improves audio quality
        # The mic will stay constantly active (not blinking) like the fallback method
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            blocksize=chunk_samples
        ) as stream:
            chunks_recorded = 0
            
            while chunks_recorded < max_chunks:
                # Read a chunk from the continuous stream
                chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("⚠️ Audio buffer overflow detected")
                
                # Flatten to 1D array
                chunk = chunk.flatten()
                
                # Calculate energy (RMS) of the chunk
                energy = np.sqrt(np.mean(chunk**2))
                max_energy_seen = max(max_energy_seen, energy)
                
                # Store the chunk
                all_audio_chunks.append(chunk)
                chunks_recorded += 1
                
                # Analyze for speech vs silence
                if energy > silence_threshold:
                    # Speech detected - mark that we've heard speech
                    has_speech = True
                    silent_chunk_count = 0
                else:
                    # Silence detected
                    if has_speech:
                        # We've had speech before, so count this silence AFTER speech
                        silent_chunk_count += 1
                        
                        # If we've had enough silence after speech, stop recording
                        if silent_chunk_count >= silence_chunks:
                            print(f"✅ Detected end of speech. Processing... (Max energy: {max_energy_seen:.4f})")
                            break
                    # CRITICAL: If no speech yet, ignore ALL initial silence
                    # Keep listening indefinitely until user actually speaks
                    # Do NOT count silence before speech - this prevents false stops
        
        # CRITICAL: Never transcribe if no speech was detected
        # This prevents Whisper from hallucinating text from background noise
        if not has_speech:
            raise Exception(
                f"No speech detected (max energy: {max_energy_seen:.4f}, threshold: {silence_threshold}). "
                f"Please speak to continue. The system will keep listening until you speak."
            )
        
        # Concatenate all chunks
        if all_audio_chunks:
            audio_data = np.concatenate(all_audio_chunks)
            audio_data = audio_data.flatten()
            
            # Check if we have meaningful audio
            if len(audio_data) < sample_rate * 0.5:
                raise Exception(f"Recording too short ({len(audio_data)/sample_rate:.2f}s). Please speak longer.")
            
            # Check audio level before trimming
            pre_trim_rms = np.sqrt(np.mean(audio_data**2))
            print(f"Before trimming: {len(audio_data)/sample_rate:.2f}s, RMS: {pre_trim_rms:.4f}")
            
            # Trim trailing silence (but be less aggressive)
            audio_data = trim_silence(audio_data, sample_rate, silence_threshold * 0.5)
            
            # Final check
            if len(audio_data) < sample_rate * 0.3:
                raise Exception("Audio too short after processing. Please try speaking louder or longer.")
            
            post_trim_rms = np.sqrt(np.mean(audio_data**2))
            print(f"After trimming: {len(audio_data)/sample_rate:.2f}s, RMS: {post_trim_rms:.4f}")
            
            return audio_data, sample_rate
        else:
            raise Exception("No audio recorded.")
            
    except Exception as e:
        raise Exception(f"Failed to record audio: {str(e)}. Please check microphone permissions.")

def trim_silence(audio_data: np.ndarray, sample_rate: int, threshold: float, chunk_size: int = 1600) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.
    
    Args:
        audio_data: Audio data array
        sample_rate: Sample rate
        threshold: Energy threshold for silence
        chunk_size: Size of chunks to analyze
    
    Returns:
        Trimmed audio data
    """
    # Find first non-silent chunk
    start_idx = 0
    for i in range(0, len(audio_data) - chunk_size, chunk_size):
        chunk = audio_data[i:i + chunk_size]
        energy = np.sqrt(np.mean(chunk**2))
        if energy > threshold:
            start_idx = max(0, i - chunk_size)  # Include a bit before for natural start
            break
    
    # Find last non-silent chunk
    end_idx = len(audio_data)
    for i in range(len(audio_data) - chunk_size, 0, -chunk_size):
        chunk = audio_data[i:i + chunk_size]
        energy = np.sqrt(np.mean(chunk**2))
        if energy > threshold:
            end_idx = min(len(audio_data), i + chunk_size + chunk_size)  # Include a bit after
            break
    
    return audio_data[start_idx:end_idx]

def record_and_transcribe(use_silence_detection: bool = True, **kwargs) -> str:
    """
    Convenience function: Record audio and transcribe in one call.
    
    Args:
        use_silence_detection: If True, use automatic silence detection (default)
        **kwargs: Additional arguments for recording functions
            - For silence detection: silence_threshold, silence_duration, max_duration
            - For fixed duration: duration
    
    Returns:
        Transcribed text string
    """
    if use_silence_detection:
        audio_data, sample_rate = record_audio_with_silence_detection(**kwargs)
    else:
        duration = kwargs.get('duration', 5.0)
        audio_data, sample_rate = record_audio(duration=duration)
    
    return transcribe_audio(audio_data, sample_rate=sample_rate)


# ---------------------------
# Text-to-Speech (TTS) Functions
# ---------------------------

def speak_text_edge(text: str, rate: int = 200) -> None:
    """
    Speak text using Edge TTS (Microsoft Edge voices).
    High-quality, natural-sounding speech.
    
    Args:
        text: Text to speak
        rate: Speech rate in words per minute (default: 200)
        Note: Edge TTS rate is in percentage, we convert WPM to percentage
    
    Raises:
        Exception: If speech fails or edge-tts is not available
    """
    try:
        import edge_tts
        import subprocess
        import platform
        import asyncio
    except ImportError:
        raise Exception("edge-tts is not installed. Install with: pip install edge-tts")
    
    if not text or len(text.strip()) == 0:
        return
    
    try:
        # Edge TTS uses percentage for rate (default is +0%, range is -50% to +100%)
        # Convert WPM to percentage: 200 WPM = 0%, 100 WPM = -50%, 300 WPM = +50%
        # Formula: rate_percent = ((rate - 200) / 200) * 100, clamped to [-50, 100]
        rate_percent = ((rate - 200) / 200) * 100
        rate_percent = max(-50, min(100, rate_percent))
        
        # Use a good default English voice (en-US-AriaNeural is high quality)
        voice = "en-US-AriaNeural"
        
        print(f"🔊 Edge TTS: Generating audio for text (length: {len(text)} chars, rate: {rate} WPM = {rate_percent:+.0f}%)")
        
        # Create temporary file path
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Edge TTS uses async/await, so we need to run it in an async context
        async def generate_and_save():
            communicate = edge_tts.Communicate(text, voice, rate=f"{rate_percent:+.0f}%")
            await communicate.save(tmp_path)
        
        # Run the async function
        print(f"🔊 Edge TTS: Saving audio to {tmp_path}")
        asyncio.run(generate_and_save())
        
        # Verify file was created
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise Exception(f"Edge TTS failed to generate audio file: {tmp_path}")
        
        print(f"✅ Edge TTS: Audio file generated ({os.path.getsize(tmp_path)} bytes)")
        
        # Play audio using system player
        # Note: Edge TTS generates MP3 files, so we convert to WAV for reliable playback
        system = platform.system()
        if system == "Darwin":  # macOS
            # Convert MP3 to WAV and use afplay (more reliable than ffplay)
            wav_path = tmp_path.replace('.mp3', '.wav')
            print(f"🔊 Edge TTS: Converting MP3 to WAV: {wav_path}")
            try:
                # Convert MP3 to WAV using ffmpeg
                result = subprocess.run(
                    ["ffmpeg", "-i", tmp_path, "-y", wav_path],
                    check=True,
                    timeout=30,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE  # Capture stderr to see errors if conversion fails
                )
                print(f"✅ Edge TTS: Conversion successful")
                
                # Verify WAV file was created
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                    raise Exception(f"ffmpeg conversion failed: WAV file is empty or missing")
                
                # Play WAV using native macOS afplay (always works)
                print(f"🔊 Edge TTS: Playing audio with afplay")
                subprocess.run(["afplay", wav_path], check=True, timeout=300)
                print(f"✅ Edge TTS: Playback completed")
                
                # Clean up WAV file
                try:
                    os.unlink(wav_path)
                except:
                    pass
            except FileNotFoundError:
                raise Exception("ffmpeg is not installed. Please install: brew install ffmpeg")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                print(f"❌ Edge TTS: Conversion/playback failed: {error_msg}")
                raise Exception(f"Failed to convert/play audio: {error_msg}")
        elif system == "Linux":
            # Try mpg123 first, fallback to ffplay
            try:
                subprocess.run(["mpg123", "-q", tmp_path], check=True, timeout=300)
            except (FileNotFoundError, subprocess.CalledProcessError):
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path], check=True, timeout=300)
        elif system == "Windows":
            subprocess.run(["start", tmp_path], shell=True, check=True, timeout=300)
        else:
            # Fallback: try to use default player
            subprocess.run(["open", tmp_path] if system == "Darwin" else ["xdg-open", tmp_path], check=True, timeout=300)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
            
    except subprocess.TimeoutExpired:
        raise Exception("Speech synthesis timed out")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to play audio: {e}")
    except Exception as e:
        raise Exception(f"Failed to speak text with Edge TTS: {e}")


def speak_text_macos(text: str, voice: Optional[str] = None, rate: int = 200) -> None:
    """
    Speak text using macOS 'say' command.
    
    Args:
        text: Text to speak
        voice: Optional voice name (e.g., "Alex", "Samantha", "Victoria")
        rate: Speech rate in words per minute (default: 200)
    
    Raises:
        Exception: If speech fails
    """
    import subprocess
    
    if not text or len(text.strip()) == 0:
        return
    
    # Clean text for command line (escape special characters)
    text_clean = text.replace('"', '\\"').replace('$', '\\$')
    
    # Build command
    cmd = ['say']
    if voice:
        cmd.extend(['-v', voice])
    cmd.extend(['-r', str(rate)])
    cmd.append(text_clean)
    
    try:
        subprocess.run(cmd, check=True, timeout=300)  # 5 minute timeout
    except subprocess.TimeoutExpired:
        raise Exception("Speech synthesis timed out")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to speak text: {e}")
    except FileNotFoundError:
        raise Exception("macOS 'say' command not found. This feature requires macOS.")


def speak_text_pyttsx3(text: str, rate: int = 200, voice_id: Optional[str] = None) -> None:
    """
    Speak text using pyttsx3 (cross-platform TTS).
    
    Args:
        text: Text to speak
        rate: Speech rate in words per minute (default: 200)
        voice_id: Optional voice ID (platform-specific)
    
    Raises:
        Exception: If speech fails or pyttsx3 is not available
    """
    if not PYTTSX3_AVAILABLE:
        raise Exception("pyttsx3 is not installed. Install with: pip install pyttsx3")
    
    if not text or len(text.strip()) == 0:
        return
    
    try:
        engine = pyttsx3.init()
        
        # Set speech rate
        engine.setProperty('rate', rate)
        
        # Set voice if specified
        if voice_id:
            voices = engine.getProperty('voices')
            for voice in voices:
                if voice_id in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        raise Exception(f"Failed to speak text with pyttsx3: {e}")


def get_available_voices() -> List[Tuple[str, str]]:
    """
    Get list of available macOS voices (English only).
    
    Returns:
        List of tuples: (voice_name, description) for English voices
        Example: [("Alex", "Alex (en_US)"), ("Samantha", "Samantha (en_US)")]
    
    Raises:
        Exception: If voice listing fails or not on macOS
    """
    import platform
    import subprocess
    
    if platform.system() != "Darwin":
        return []  # Not macOS, return empty list
    
    try:
        # Get all voices
        result = subprocess.run(
            ['say', '-v', '?'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        
        voices = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            
            # Parse line format: "VoiceName              lang    # Description"
            parts = line.split('#')
            if len(parts) < 2:
                continue
            
            voice_info = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract voice name and language
            voice_parts = voice_info.split()
            if len(voice_parts) < 2:
                continue
            
            voice_name = voice_parts[0]
            lang_code = voice_parts[1] if len(voice_parts) > 1 else ""
            
            # Filter for English voices only
            if lang_code in ['en_US', 'en_GB']:
                # Clean up voice name (remove language suffix if present)
                clean_name = voice_name.split('(')[0].strip()
                display_name = f"{clean_name} ({lang_code})"
                voices.append((clean_name, display_name))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_voices = []
        for voice_name, display_name in voices:
            if voice_name not in seen:
                seen.add(voice_name)
                unique_voices.append((voice_name, display_name))
        
        return sorted(unique_voices, key=lambda x: x[0])  # Sort by voice name
        
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        return []
    except Exception as e:
        # Return empty list on any error (graceful degradation)
        return []


def speak_text(text: str, method: str = "auto", voice: Optional[str] = None, rate: int = 200) -> None:
    """
    Speak text using the best available TTS method.
    Primary: Edge TTS (high quality, cross-platform)
    Fallback: macOS 'say' command (if Edge TTS fails)
    
    Args:
        text: Text to speak
        method: TTS method to use ("auto", "edge", "macos", "pyttsx3")
        voice: Optional voice name/ID (ignored for Edge TTS, uses default high-quality voice)
        rate: Speech rate in words per minute (default: 200)
    
    Raises:
        Exception: If speech fails
    """
    import platform
    
    # Clean text for speech (remove markdown, code blocks, etc.)
    cleaned_text = clean_text_for_speech(text)
    if not cleaned_text:
        return  # Nothing to speak after cleaning
    
    # Determine method
    if method == "auto":
        # Try Edge TTS first (best quality)
        try:
            print("🔊 Using Edge TTS for speech synthesis...")
            speak_text_edge(cleaned_text, rate=rate)
            print("✅ Edge TTS completed successfully")
            return
        except Exception as e:
            # Fallback to macOS say if Edge TTS fails
            print(f"⚠️ Edge TTS failed: {e}, falling back to macOS 'say'...")
            if platform.system() == "Darwin":  # macOS
                try:
                    speak_text_macos(cleaned_text, voice=voice, rate=rate)
                    print("✅ macOS 'say' completed successfully")
                    return
                except Exception as fallback_error:
                    raise Exception(f"TTS failed. Edge TTS error: {e}, macOS fallback error: {fallback_error}")
            else:
                raise Exception(f"TTS failed. Edge TTS error: {e}")
    
    # Call specific method if requested
    if method == "edge":
        speak_text_edge(cleaned_text, rate=rate)
    elif method == "macos":
        speak_text_macos(cleaned_text, voice=voice, rate=rate)
    elif method == "pyttsx3":
        speak_text_pyttsx3(cleaned_text, rate=rate, voice_id=voice)
    else:
        raise Exception(f"Unknown TTS method: {method}")

