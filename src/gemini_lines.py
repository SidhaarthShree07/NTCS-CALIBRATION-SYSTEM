import base64
import os
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import io
import json
import json

def get_gemini_line_estimates(image_path, api_key, road_polygon=None):
    """
    Given an image path (screenshot of first model frame with mask and polygon overlay),
    call Gemini via LangChain to estimate optimal y-coordinates for line_A_y and line_B_y,
    and also estimate the road width in meters.
    
    Args:
        image_path: Path to screenshot image
        api_key: Gemini API key
        road_polygon: Optional numpy array of polygon points to extract bounds
        
    Returns: dict with 'line_A_y', 'line_B_y', 'road_width_m', and 'note'.
    """
    print(f"[Gemini] Starting with image_path: {image_path}")
    print(f"[Gemini] API key provided: {bool(api_key)}")
    
    try:
        # Read and encode image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image dimensions
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            img_width, img_height = img.size
        print(f"[Gemini] Image dimensions: {img_width}x{img_height}")
        
        # Extract polygon bounds if available
        polygon_top_y = None
        polygon_bottom_y = None
        if road_polygon is not None and len(road_polygon) > 0:
            import numpy as np
            polygon_top_y = int(np.min(road_polygon[:, 1]))
            polygon_bottom_y = int(np.max(road_polygon[:, 1]))
            print(f"[Gemini] Road polygon Y-range: {polygon_top_y} → {polygon_bottom_y}")
        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        print(f"[Gemini] Image loaded: {len(img_bytes)} bytes")
        
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"[Gemini] Image encoded to base64: {len(img_b64)} chars")
    except Exception as e:
        print(f"[Gemini] Error loading/encoding image: {e}")
        return {"line_A_y": None, "line_B_y": None, "note": f"Image loading error: {e}"}

    # System prompt: STRUCTURED + CONCISE to minimize reasoning tokens
    # Use actual polygon bounds if available for more accurate placement
    if polygon_top_y is not None and polygon_bottom_y is not None:
        # Use actual polygon range
        suggested_line_a = polygon_top_y + int((polygon_bottom_y - polygon_top_y) * 0.15)  # 15% from top
        suggested_line_b = polygon_top_y + int((polygon_bottom_y - polygon_top_y) * 0.75)  # 75% from top
        system_prompt = (
            f"Image analysis task: Place 2 horizontal lines on {img_width}x{img_height}px road image AND estimate road width.\n"
            f"Green polygon Y-range: {polygon_top_y} (far/top) to {polygon_bottom_y} (near/bottom).\n\n"
            f"Requirements:\n"
            f"- line_A_y: Entry line near Y={suggested_line_a} (upper 15-25% of green area)\n"
            f"- line_B_y: Exit line near Y={suggested_line_b} (lower 70-80% of green area)\n"
            f"- Constraint: line_A_y < line_B_y (A above B)\n"
            f"- Both inside green polygon (between {polygon_top_y} and {polygon_bottom_y})\n"
            f"- Maximize separation for accuracy\n"
            f"- road_width_m: Estimate real-world width of the visible road in METERS based on the number of lanes and road type. Count the lanes and multiply by ~3.5m per lane. Add shoulders if visible.\n\n"
            f"Output JSON only: {{\"line_A_y\": <int>, \"line_B_y\": <int>, \"road_width_m\": <float>, \"note\": \"<text>\"}}"
        )
    else:
        # Fallback: use image height estimates
        system_prompt = (
            f"Image analysis task: Place 2 horizontal lines on {img_width}x{img_height}px road image AND estimate road width.\n"
            f"Green polygon = road area. Y-axis: 0=top, {img_height}=bottom.\n\n"
            f"Requirements:\n"
            f"- line_A_y: Entry line (upper section of green area)\n"
            f"- line_B_y: Exit line (lower section of green area)\n"
            f"- Constraint: line_A_y < line_B_y (A above B)\n"
            f"- Separation: {int(img_height * 0.3)}-{int(img_height * 0.5)}px apart\n"
            f"- Both inside green polygon with margins\n"
            f"- road_width_m: Estimate real-world width of the visible road in METERS based on the number of lanes and road type. Count the lanes and multiply by ~3.5m per lane. Add shoulders if visible.\n\n"
            f"Output JSON only: {{\"line_A_y\": <int>, \"line_B_y\": <int>, \"road_width_m\": <float>, \"note\": \"<text>\"}}"
        )

    # Prepare LangChain Gemini chat
    try:
        # Log the exact prompt that will be sent for transparency
        print("[Gemini] Prompt (lines):\n" + system_prompt)
        print(f"[Gemini] Initializing with API key: {api_key[:20]}...")
        chat = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",  # Using 2.5 flash which supports images
            google_api_key=api_key,
            temperature=0.3,  # Slightly higher to reduce deterministic overthinking
            max_output_tokens=4096,  # Higher budget for reasoning spikes
        )
        print("[Gemini] Model initialized successfully")
        
        # Send image as base64 data URL in HumanMessage (LangChain multimodal format)
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Return JSON with line positions:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                }
            ]
        )
        print(f"[Gemini] Sending request with image ({len(img_b64)} chars base64)")
        
        response = chat.invoke([
            SystemMessage(content=system_prompt + "\n\nConstraints: Respond with JSON only. No markdown, no code fences, no explanations."),
            human_message
        ])
        print(f"[Gemini] Response received: {type(response)}")
        print(f"[Gemini] Response object: {response}")
        
        # Log raw Gemini response for debugging
        text = response.content if hasattr(response, 'content') else str(response)
        print(f"[Gemini raw response] '{text}'")
        # Capture finish reason if available
        finish_reason = ''
        try:
            finish_reason = response.response_metadata.get('finish_reason', '')
        except Exception:
            finish_reason = ''
    except Exception as api_error:
        print(f"[Gemini API Error] {type(api_error).__name__}: {api_error}")
        print(f"[Gemini API Error] Full traceback:")
        traceback.print_exc()
        return {"line_A_y": None, "line_B_y": None, "note": f"Gemini API error: {api_error}"}
    # Extract JSON from response
    try:
        if not text or not text.strip():
            raise ValueError("Gemini returned empty response.")
        
        # Remove markdown code block markers if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            # Remove ```json at start and ``` at end
            lines = cleaned_text.splitlines()
            cleaned_text = "\n".join(lines[1:])  # Remove first line with ```json
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]  # Remove closing ```
        elif cleaned_text.startswith("```"):
            # Remove generic ``` markers
            cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        # Try to find JSON object
        jmatch = None
        if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
            jmatch = cleaned_text
        else:
            # Try to extract JSON from text
            for line in cleaned_text.splitlines():
                if line.strip().startswith('{'):
                    jmatch = line.strip()
                    break
            if not jmatch and '{' in cleaned_text and '}' in cleaned_text:
                jmatch = cleaned_text[cleaned_text.find('{'):cleaned_text.rfind('}')+1]
        
        if not jmatch:
            raise ValueError(f"No JSON object found in Gemini response: {cleaned_text}")
        
        result = json.loads(jmatch)
        print(f"[Gemini] Successfully parsed: {result}")
        # If model ran out of tokens or JSON seems incomplete, attempt one compact retry
        if finish_reason == 'MAX_TOKENS':
            raise ValueError("Response truncated (MAX_TOKENS)")
        
        # Validation step: Ask Gemini to confirm the placement
        line_a = result.get('line_A_y')
        line_b = result.get('line_B_y')
        
        if line_a is not None and line_b is not None:
            # Validate the coordinate order
            if line_a >= line_b:
                print(f"[Gemini] WARNING: Incorrect coordinate order (A={line_a}, B={line_b}). A should be < B.")
                print(f"[Gemini] Swapping coordinates to correct order...")
                result['line_A_y'], result['line_B_y'] = line_b, line_a
                result['note'] = f"Coordinates auto-corrected. {result.get('note', '')}"
            
            # Ask Gemini to validate the placement (OPTIONAL - skip if issues)
            # Commenting out validation to avoid token limit issues
            """
            validation_prompt = (
                f"I placed calibration lines at:\n"
                f"- line_A_y = {result['line_A_y']} (entry line, top)\n"
                f"- line_B_y = {result['line_B_y']} (exit line, bottom)\n"
                f"- Separation = {abs(result['line_B_y'] - result['line_A_y'])} pixels\n\n"
                f"Image dimensions: {img_width}x{img_height}\n\n"
                f"Is this placement optimal for speed detection? Are both lines inside the road polygon?\n"
                f"Respond with JSON: {{\"valid\": true/false, \"feedback\": \"brief note\"}}"
            )
            
            try:
                validation_msg = HumanMessage(content=[
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ])
                validation_response = chat.invoke([validation_msg])
                validation_text = validation_response.content if hasattr(validation_response, 'content') else str(validation_response)
                print(f"[Gemini Validation] {validation_text}")
                
                # Try to parse validation response
                validation_cleaned = validation_text.strip()
                if validation_cleaned.startswith("```json"):
                    validation_cleaned = "\n".join(validation_cleaned.splitlines()[1:])
                    if validation_cleaned.endswith("```"):
                        validation_cleaned = validation_cleaned[:-3]
                validation_cleaned = validation_cleaned.strip()
                
                if '{' in validation_cleaned and '}' in validation_cleaned:
                    val_json = validation_cleaned[validation_cleaned.find('{'):validation_cleaned.rfind('}')+1]
                    validation_result = json.loads(val_json)
                    if not validation_result.get('valid', True):
                        result['note'] = f"[VALIDATION CONCERN] {validation_result.get('feedback', 'Check line placement')}. Original: {result.get('note', '')}"
                    else:
                        result['note'] = f"✓ Validated. {validation_result.get('feedback', '')} {result.get('note', '')}"
                    print(f"[Gemini] Validation result: {validation_result}")
            except Exception as val_error:
                print(f"[Gemini] Validation step failed: {val_error}")
                # Continue with original result if validation fails
            """
            print(f"[Gemini] Skipping validation step to avoid token issues")
            result['note'] = f"✓ Placement accepted. {result.get('note', '')}"
        
        # Validate road width - accept Gemini's estimate with light sanity check
        road_width = result.get('road_width_m')
        if road_width is not None:
            road_width = float(road_width)
            if road_width < 3.0:
                print(f"[Gemini] Road width {road_width}m unrealistically small, setting to 7m (single lane + shoulders)")
                result['road_width_m'] = 7.0
                result['note'] = f"Road width adjusted (was {road_width:.1f}m). {result.get('note', '')}"
            elif road_width > 60.0:
                print(f"[Gemini] Road width {road_width}m unrealistically large, capping at 50m")
                result['road_width_m'] = 50.0
                result['note'] = f"Road width capped (was {road_width:.1f}m). {result.get('note', '')}"
            else:
                print(f"[Gemini] Road width estimated: {road_width}m")
        else:
            # If no road width provided, use a reasonable default
            print(f"[Gemini] No road width provided, using default 10.0m")
            result['road_width_m'] = 10.0
            result['note'] = f"Road width defaulted to 10m. {result.get('note', '')}"
        
        return result
    except Exception as e:
        print(f"[Gemini error] Could not parse response: {e}")
        # Retry once with a stricter, compact JSON-only request
        try:
            print("[Gemini] Retrying with compact JSON-only request...")
            retry_message = HumanMessage(
                content=[
                    {"type": "text", "text": (
                        "Return ONLY compact JSON (no markdown, no code fences). Keys: "
                        "{\"line_A_y\": <int>, \"line_B_y\": <int>, \"note\": \"<text>\"}."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            )
            response2 = chat.invoke([
                SystemMessage(content=system_prompt + "\n\nOutput strictly JSON with keys line_A_y, line_B_y, note. No extra text."),
                retry_message
            ])
            text2 = response2.content if hasattr(response2, 'content') else str(response2)
            print(f"[Gemini raw response - retry] '{text2}'")
            cleaned2 = text2.strip()
            if cleaned2.startswith("```json"):
                lines = cleaned2.splitlines()
                cleaned2 = "\n".join(lines[1:])
                if cleaned2.endswith("```"):
                    cleaned2 = cleaned2[:-3]
            elif cleaned2.startswith("```"):
                cleaned2 = cleaned2[3:]
                if cleaned2.endswith("```"):
                    cleaned2 = cleaned2[:-3]
            cleaned2 = cleaned2.strip()
            if cleaned2.startswith('{') and cleaned2.endswith('}'):
                j2 = cleaned2
            elif '{' in cleaned2 and '}' in cleaned2:
                j2 = cleaned2[cleaned2.find('{'):cleaned2.rfind('}')+1]
            else:
                raise ValueError("Retry response missing JSON object")
            result2 = json.loads(j2)
            print(f"[Gemini] Successfully parsed (retry): {result2}")
            # Validate order
            la = result2.get('line_A_y')
            lb = result2.get('line_B_y')
            if la is not None and lb is not None and la >= lb:
                print(f"[Gemini] WARNING: Incorrect order on retry (A={la}, B={lb}). Swapping.")
                result2['line_A_y'], result2['line_B_y'] = lb, la
                result2['note'] = f"Coordinates auto-corrected. {result2.get('note', '')}"
            return result2
        except Exception as e2:
            print(f"[Gemini error] Retry failed: {e2}")
            # Fallback to alternate models if available
            fallback_models = ["gemini-3-flash-preview"]
            for m in fallback_models:
                try:
                    print(f"[Gemini] Fallback model attempt: {m}")
                    chat_fb = ChatGoogleGenerativeAI(
                        model=m,
                        google_api_key=api_key,
                        temperature=0.3,
                        max_output_tokens=4096,
                    )
                    resp_fb = chat_fb.invoke([
                        SystemMessage(content=system_prompt + "\n\nConstraints: Respond with JSON only. No markdown, no code fences, no explanations."),
                        human_message
                    ])
                    text_fb = resp_fb.content if hasattr(resp_fb, 'content') else str(resp_fb)
                    print(f"[Gemini raw response - {m}] '{text_fb}'")
                    cleaned_fb = text_fb.strip()
                    if cleaned_fb.startswith("```json"):
                        lines = cleaned_fb.splitlines()
                        cleaned_fb = "\n".join(lines[1:])
                        if cleaned_fb.endswith("```"):
                            cleaned_fb = cleaned_fb[:-3]
                    elif cleaned_fb.startswith("```"):
                        cleaned_fb = cleaned_fb[3:]
                        if cleaned_fb.endswith("```"):
                            cleaned_fb = cleaned_fb[:-3]
                    cleaned_fb = cleaned_fb.strip()
                    if cleaned_fb.startswith('{') and cleaned_fb.endswith('}'):
                        jfb = cleaned_fb
                    elif '{' in cleaned_fb and '}' in cleaned_fb:
                        jfb = cleaned_fb[cleaned_fb.find('{'):cleaned_fb.rfind('}')+1]
                    else:
                        raise ValueError("Fallback response missing JSON object")
                    res_fb = json.loads(jfb)
                    print(f"[Gemini] Successfully parsed (fallback {m}): {res_fb}")
                    # Validate order
                    la = res_fb.get('line_A_y')
                    lb = res_fb.get('line_B_y')
                    if la is not None and lb is not None and la >= lb:
                        print(f"[Gemini] WARNING: Incorrect order on fallback (A={la}, B={lb}). Swapping.")
                        res_fb['line_A_y'], res_fb['line_B_y'] = lb, la
                        res_fb['note'] = f"Coordinates auto-corrected. {res_fb.get('note', '')}"
                    return res_fb
                except Exception as e_fb:
                    print(f"[Gemini error] Fallback model {m} failed: {e_fb}")

            # If all fallbacks fail, return parse error
            return {"line_A_y": None, "line_B_y": None, "note": f"Gemini parsing error: {e}. Raw response: {text}"}
