import re
import google.generativeai as genai
from config import GEMINI_API_KEYS, GEMINI_MODEL

current_key_index = 0
exhausted_keys = set()


def get_active_client():
    global current_key_index
    while current_key_index < len(GEMINI_API_KEYS):
        if current_key_index not in exhausted_keys:
            genai.configure(api_key=GEMINI_API_KEYS[current_key_index])
            return genai.GenerativeModel(GEMINI_MODEL)
        current_key_index += 1
    return None


def rotate_key():
    global current_key_index
    exhausted_keys.add(current_key_index)
    current_key_index += 1
    if current_key_index >= len(GEMINI_API_KEYS):
        print("[ERROR] All 5 Gemini API keys are exhausted. Please update your API keys in config.py.")
        return None
    genai.configure(api_key=GEMINI_API_KEYS[current_key_index])
    return genai.GenerativeModel(GEMINI_MODEL)


def strip_markdown(text):
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'#+\s+', '', text)
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def build_system_prompt(prediction_context):
    prediction = prediction_context['prediction']
    confidence = prediction_context['confidence']
    file_type = prediction_context['file_type']
    file_name = prediction_context.get('file_name', 'the uploaded file')

    verdict = "a DEEPFAKE (AI-generated or manipulated media)" if prediction == "Fake" else "REAL (authentic, not AI-generated)"
    confidence_desc = (
        "extremely high" if confidence >= 95 else
        "high" if confidence >= 85 else
        "moderate" if confidence >= 70 else
        "low"
    )

    if prediction == "Fake":
        artifact_hints = (
            "Common manipulation artifacts in this category include: "
            "blurring or smoothing around facial boundaries, unnatural skin texture with over-smoothed regions, "
            "inconsistent lighting direction on different face regions, misaligned eye reflections (corneal specular highlights), "
            "temporal flickering in video (face area changes between frames more than background), "
            "color bleeding at face-to-hair boundaries, and frequency-domain anomalies visible as GradCAM hotspots "
            "concentrated on cheeks, jawline, eye corners, and forehead regions."
        )
    else:
        artifact_hints = (
            "Indicators supporting authenticity: "
            "natural skin texture with visible pores and fine details, consistent lighting across the entire face, "
            "coherent corneal reflections matching the scene lighting, stable facial geometry across video frames, "
            "natural noise structure in flat regions (sky, walls) without AI smoothing artifacts, "
            "GradCAM activations distributed naturally across facial features rather than concentrated on boundary regions."
        )

    if file_type == "image":
        pipeline_details = (
            f"Pipeline used on '{file_name}': "
            f"The image was resized to 224x224 pixels, normalized using ImageNet mean [0.485, 0.456, 0.406] "
            f"and std [0.229, 0.224, 0.225], then passed through EfficientViT-B0 (timm library). "
            f"The model was trained on 190,335 images (70,001 real + 70,001 fake in training, with validation and test splits). "
            f"Training augmentations: HorizontalFlip, ColorJitter, Affine transforms. "
            f"GradCAM was applied on the last stage (model.stages[-1]) to generate the heatmap. "
            f"RGB frequency histogram was generated to show pixel intensity distribution across R, G, B channels."
        )
        rgb_analysis_hint = (
            "An RGB frequency histogram of this image has been generated. "
            "In deepfakes, the histogram often shows: unusually smooth/uniform channel distributions (due to GAN generation), "
            "shifted peaks in blue or red channels (color grading artifacts), or narrow intensity ranges "
            "compared to natural photos which show broader, multi-modal distributions. "
            "In real images, all three channels typically show natural, organic distribution curves."
        )
    else:
        pipeline_details = (
            f"Pipeline used on '{file_name}': "
            f"40 evenly-spaced frames were extracted from the video, each resized to 224x224 and normalized with ImageNet stats. "
            f"EfficientViT-B0 (video variant, drop_rate=0.3) ran inference on each frame independently. "
            f"Final prediction used average probability across all 40 frames (threshold 0.5). "
            f"The top 5 most confident frames (furthest from 0.5 probability) were selected for GradCAM analysis. "
            f"The model was trained on 66 videos (33 AI-generated + 33 real), achieving 94.95% test accuracy. "
            f"An RGB histogram was generated from the most representative frame for spectral analysis."
        )
        rgb_analysis_hint = (
            "An RGB histogram from the most confident frame has been generated. "
            "In deepfake videos, per-frame histograms may show: temporal inconsistency (histogram shape changes drastically between frames), "
            "narrow blue-channel peaks from GAN post-processing, or unnatural uniformity. "
            "Real video frames show natural shot-to-shot variation with organic multi-modal RGB distributions."
        )

    system_prompt = (
        f"You are DeepGuard AI, the expert forensic analysis assistant inside a deepfake detection platform. "
        f"You specialize in explaining AI model decisions to both technical and non-technical users. "
        f"You have just completed a full forensic analysis. Here are the exact results:\n\n"
        f"File: {file_name}\n"
        f"File type: {file_type}\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.2f}% ({confidence_desc} confidence)\n\n"
        f"=== TECHNICAL PIPELINE ===\n"
        f"{pipeline_details}\n\n"
        f"=== MODEL SPECIFICATIONS ===\n"
        f"Architecture: EfficientViT-B0 (from timm library, Linear Transformer variant optimized for efficiency). "
        f"Loss: BCEWithLogitsLoss with pos_weight. Optimizer: AdamW with weight decay 1e-4. "
        f"Scheduler: CosineAnnealingLR. Hardware: Tesla P100-PCIE-16GB GPU. "
        f"Binary classification: sigmoid output >= 0.5 = Real, < 0.5 = Fake.\n\n"
        f"=== ARTIFACT ANALYSIS CONTEXT ===\n"
        f"{artifact_hints}\n\n"
        f"=== RGB SPECTRAL ANALYSIS CONTEXT ===\n"
        f"{rgb_analysis_hint}\n\n"
        f"=== RESPONSE RULES ===\n"
        f"You MUST follow these rules strictly:\n"
        f"1. Always reference the exact confidence score ({confidence:.2f}%), model name (EfficientViT-B0), and verdict ({prediction}) in every answer.\n"
        f"2. For simple factual questions (what is the prediction, confidence, etc.), give 2-3 sentences.\n"
        f"3. For technical questions (how does GradCAM work, what artifacts were found, what does the RGB graph show), give 4-5 sentences with specific technical details from the context above.\n"
        f"4. When explaining fake detection, describe specific visual artifact cues (boundary anomalies, lighting inconsistencies, frequency patterns).\n"
        f"5. When explaining real detection, describe what natural features supported the classification.\n"
        f"6. If asked about the RGB histogram, explain what the channel distributions reveal about the media's authenticity.\n"
        f"7. NEVER use markdown, bullet points, asterisks, headers, or numbered lists. Write in natural flowing sentences only.\n"
        f"8. NEVER ask the user clarifying questions. Always give a direct, substantive answer.\n"
        f"9. If asked about preprocessing, cite exact values: 224x224 resize, ImageNet normalization [0.485,0.456,0.406] mean.\n"
        f"10. Vary sentence structure. Do not begin every response the same way.\n"
        f"11. If asked what the user should do with a fake result, provide practical advice: report to platform, preserve evidence, use reverse image search, check metadata.\n"
        f"12. Be confident but honest about model limitations — no model is perfect, and this one is trained for research purposes."
    )
    return system_prompt


def chat_with_gemini(messages, prediction_context):
    model = get_active_client()
    if model is None:
        return "All API keys are exhausted. Please update your Gemini API keys."

    system_prompt = build_system_prompt(prediction_context)

    try:
        conversation = [
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": [
                f"Understood. I have completed the forensic analysis on {prediction_context['file_name']}. "
                f"The result is {prediction_context['prediction']} with a confidence of {prediction_context['confidence']:.2f}%. "
                f"I have full context on the EfficientViT-B0 pipeline, GradCAM heatmap, and RGB frequency distribution. "
                f"Ready to provide detailed forensic explanations."
            ]},
        ]

        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            conversation.append({"role": role, "parts": [msg["message"]]})

        response = model.generate_content(conversation)
        return strip_markdown(response.text)

    except Exception as e:
        if "quota" in str(e).lower() or "exhausted" in str(e).lower() or "429" in str(e):
            model = rotate_key()
            if model is None:
                return "All API keys are exhausted. Please update your Gemini API keys."
            try:
                response = model.generate_content(conversation)
                return strip_markdown(response.text)
            except Exception:
                return "Unable to process request. Please try again later."
        return f"Error communicating with Gemini: {str(e)}"