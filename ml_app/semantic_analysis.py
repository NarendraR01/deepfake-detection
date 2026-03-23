# semantic_analysis.py
# Place this file inside: Django Application/ml_app/semantic_analysis.py

import os
import subprocess

def extract_audio_from_video(video_path, output_audio_path=None):
    """Extract audio from video using ffmpeg."""
    if output_audio_path is None:
        base = os.path.splitext(video_path)[0]
        output_audio_path = base + "_audio.wav"
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_audio_path
    except Exception as e:
        print(f"[SemanticAnalysis] Audio extraction failed: {e}")
        return None


def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper (tiny model)."""
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, fp16=False)
        transcript = result.get("text", "").strip()
        return transcript if transcript else None
    except Exception as e:
        print(f"[SemanticAnalysis] Transcription failed: {e}")
        return None


def analyze_sentiment(text):
    """Analyze sentiment using DistilBERT."""
    try:
        from transformers import pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        result = sentiment_pipeline(text[:512])[0]
        label = result["label"]
        score = round(result["score"] * 100, 2)
        return {"label": label, "score": score}
    except Exception as e:
        print(f"[SemanticAnalysis] Sentiment analysis failed: {e}")
        return {"label": "UNKNOWN", "score": 0.0}


def get_semantic_consistency(sentiment_label, prediction_label):
    """
    Cross-modal consistency between visual prediction and speech sentiment.

    Status values:
      consistent  → both signals agree
      doubtful    → signals contradict each other (strong mismatch)
      cautious    → partial info, uncertain

    Matrix:
      FAKE  + NEGATIVE  → consistent  (manipulation + distressing speech align)
      FAKE  + POSITIVE  → doubtful    (face swapped but speech sounds genuine — classic deepfake signature)
      REAL  + POSITIVE  → consistent  (authentic video, positive speech)
      REAL  + NEGATIVE  → doubtful    (looks real but speech is distressing — suspicious mismatch)
      NO_FACE + NEGATIVE → doubtful   (no face, concerning speech)
      NO_FACE + POSITIVE → cautious   (no face, audio only, seems ok)
    """

    if prediction_label == "NO_FACE":
        if sentiment_label == "NEGATIVE":
            return {
                "status": "doubtful",
                "note": "Face visibility was insufficient for visual deepfake analysis — this can happen when a face is partially visible, at an extreme angle, obscured, or not present at all in this video. However, the speech sentiment is strongly negative — audio content raises concern. Manual review is recommended."
            }
        elif sentiment_label == "POSITIVE":
            return {
                "status": "cautious",
                "note": "Face visibility was insufficient for visual deepfake analysis — this may be due to a partially visible face, poor lighting, an obscured face, or simply no face in the video. Speech sentiment appears positive. The result here is based on audio only and should be treated with caution."
            }
        else:
            return {
                "status": "cautious",
                "note": "Face visibility was insufficient for visual analysis — the face may be partially visible, at an extreme angle, too small, or absent entirely. Audio-only semantic analysis was performed. A conclusive deepfake verdict could not be reached for this video."
            }

    elif prediction_label == "FAKE":
        if sentiment_label == "NEGATIVE":
            return {
                "status": "consistent",
                "note": "Visual manipulation detected and speech sentiment is negative — both signals align. High likelihood of manipulated distress or propaganda content."
            }
        elif sentiment_label == "POSITIVE":
            return {
                "status": "doubtful",
                "note": "Cross-modal inconsistency detected — the face appears manipulated but the speech sentiment is positive and genuine-sounding. This mismatch is a strong deepfake signature: the original audio was likely retained while only the face was swapped."
            }
        else:
            return {
                "status": "cautious",
                "note": "Visual manipulation detected. Speech sentiment could not be clearly determined. Treat result as suspicious."
            }

    elif prediction_label == "REAL":
        if sentiment_label == "POSITIVE":
            return {
                "status": "consistent",
                "note": "Speech sentiment is consistent with authentic video content. Both visual and audio signals appear genuine — no cross-modal inconsistency found."
            }
        elif sentiment_label == "NEGATIVE":
            return {
                "status": "doubtful",
                "note": "Cross-modal inconsistency noted — the video appears visually authentic but the speech carries a strongly negative sentiment. This may indicate selectively edited audio or dubious context worth reviewing."
            }
        else:
            return {
                "status": "cautious",
                "note": "Video appears visually authentic. Speech sentiment could not be clearly determined. No strong cross-modal inconsistency detected."
            }

    return {
        "status": "cautious",
        "note": "Semantic consistency could not be fully determined."
    }


def run_semantic_analysis(video_path, prediction_label="UNKNOWN"):
    """
    Full semantic analysis pipeline.
    Always runs even when no face is detected — audio is always analyzed.

    Args:
        video_path:        full path to the uploaded video file
        prediction_label:  "REAL", "FAKE", or "NO_FACE"

    Returns dict with keys:
        transcript, sentiment_label, sentiment_score,
        consistency_status, consistency_note, error
    """
    result = {
        "transcript": None,
        "sentiment_label": None,
        "sentiment_score": None,
        "consistency_status": None,
        "consistency_note": None,
        "error": None
    }

    # Step 1: Extract audio
    audio_path = extract_audio_from_video(video_path)
    if not audio_path or not os.path.exists(audio_path):
        result["error"] = "Could not extract audio from video."
        return result

    # Step 2: Transcribe
    transcript = transcribe_audio(audio_path)

    # Clean up temp audio file
    try:
        os.remove(audio_path)
    except:
        pass

    if not transcript:
        result["error"] = "No speech detected in this video."
        return result

    result["transcript"] = transcript

    # Step 3: Sentiment
    sentiment = analyze_sentiment(transcript)
    result["sentiment_label"] = sentiment["label"]
    result["sentiment_score"]  = sentiment["score"]

    # Step 4: Cross-modal consistency
    consistency = get_semantic_consistency(sentiment["label"], prediction_label)
    result["consistency_status"] = consistency["status"]
    result["consistency_note"]   = consistency["note"]

    return result