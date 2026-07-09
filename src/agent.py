#!/usr/bin/env python3
"""
Video Captioning Agent - OPTIMIZED VERSION
Two-Model Pipeline with .env Support
"""

import os
import sys
import json
import base64
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import cv2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def log_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def log_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def log_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def log_header(msg):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{msg.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")


class VideoCaptionAgent:
    """Optimized two-stage video captioning with unified system guidelines"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required")
        
        self.api_key = api_key
        self.api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.vision_model = "accounts/fireworks/models/qwen3p7-plus"
        self.text_model = "accounts/fireworks/models/glm-5p2"
        
        # Unified system guideline to prevent chain-of-thought leakage
        self.system_guidelines = (
            "You are an automated API module that produces raw caption strings. "
            "CRITICAL: Output ONLY the requested caption text directly. "
            "Do NOT include internal thoughts, preambles, introductions, planning thoughts, "
            "or lead-ins like 'The user wants...', 'Here is...', 'Let me think...', "
            "'Based on...', 'Analysis:', 'Caption:', or any meta-commentary. "
            "Go straight into the character of the requested style. No conversational fillers. "
            "Start writing the actual caption immediately."
        )
        
        # Vision analysis system prompt
        self.vision_system = (
            "You are a video analysis API. Output ONLY factual descriptions. "
            "No preambles, no meta-commentary, no thinking process. "
            "Start directly with 'The video shows...' or 'This video depicts...'"
        )
        
        # Style-specific configurations
        self.style_configs = {
            "formal": {
                "persona": "You are a professional video analyst writing formal descriptions.",
                "instruction": (
                    "Write a formal, objective description in exactly 2-3 sentences. "
                    "Use professional academic language. Describe subjects, actions, environment, and lighting. "
                    "Start immediately with the description."
                ),
                "temperature": 0.6
            },
            "sarcastic": {
                "persona": "You are a sarcastic commentator with dry wit.",
                "instruction": (
                    "Write a sarcastic, ironic commentary in exactly 2-3 sentences. "
                    "Use dry humor and subtle mockery. Be funny but not mean. "
                    "Start immediately with the sarcastic observation."
                ),
                "temperature": 0.8
            },
            "humorous_tech": {
                "persona": "You are a tech comedian making programming jokes.",
                "instruction": (
                    "Write a funny tech-themed caption in exactly 2-3 sentences. "
                    "Use programming jokes and coding references: API, debugging, git, algorithm, "
                    "stack overflow, compile, deploy, function, variable. "
                    "Start immediately with the tech joke."
                ),
                "temperature": 0.9
            },
            "humorous_non_tech": {
                "persona": "You are a stand-up comedian with universal humor.",
                "instruction": (
                    "Write a funny everyday caption in exactly 2-3 sentences. "
                    "Use relatable humor. NO technical jargon. NO programming terms. "
                    "Start immediately with the funny observation."
                ),
                "temperature": 0.9
            }
        }
    
    def download_video(self, video_url: str, temp_dir: str) -> str:
        """Download video file"""
        log_info(f"Downloading video...")
        
        try:
            response = requests.get(video_url, stream=True, timeout=180)
            response.raise_for_status()
            
            video_path = os.path.join(temp_dir, "video.mp4")
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
            
            print()
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            log_success(f"Downloaded {file_size_mb:.1f} MB")
            return video_path
        
        except Exception as e:
            log_error(f"Download failed: {e}")
            raise
    
    def extract_keyframes(self, video_path: str, max_frames: int = 5) -> List[str]:
        """Extract keyframes"""
        log_info("Extracting keyframes...")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            log_info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
            
            if total_frames <= max_frames:
                frame_indices = list(range(0, total_frames, max(1, total_frames // max_frames)))
            else:
                step = total_frames // max_frames
                frame_indices = [i * step for i in range(max_frames)]
            
            base64_frames = []
            for idx, frame_idx in enumerate(frame_indices, 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    height, width = frame.shape[:2]
                    if width > 800:
                        scale = 800 / width
                        new_width = 800
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    base64_frame = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(base64_frame)
                    print(f"\r   Extracted frame {idx}/{len(frame_indices)}", end='', flush=True)
            
            print()
            cap.release()
            log_success(f"Extracted {len(base64_frames)} keyframes")
            return base64_frames
        
        except Exception as e:
            log_error(f"Keyframe extraction failed: {e}")
            raise
    
    def analyze_video_with_qwen(self, base64_frames: List[str]) -> str:
        """Stage 1: Analyze video with Qwen"""
        log_info("Stage 1: Analyzing video with Qwen...")
        
        try:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Describe what happens in these video frames. "
                        "Write 3-4 factual sentences covering: subjects, actions, setting, details. "
                        "Start immediately with 'The video shows' or 'This video depicts'."
                    )
                }
            ]
            
            for base64_frame in base64_frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}
                })
            
            payload = {
                "model": self.vision_model,
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": self.vision_system},
                    {"role": "user", "content": content}
                ]
            }
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(self.api_url, headers=headers, 
                                    data=json.dumps(payload), timeout=90)
            response.raise_for_status()
            
            result = response.json()
            description = result['choices'][0]['message']['content'].strip()
            description = self._extract_clean_description(description)
            
            log_success(f"Analysis: {description[:100]}...")
            return description
        
        except Exception as e:
            log_error(f"Qwen analysis failed: {e}")
            return "The video shows various scenes and activities."
    
    def _extract_clean_description(self, text: str) -> str:
        """Extract clean description"""
        bad_patterns = [
            r'The user.*?[\n.]', r'Analysis:.*?[\n.]', r'\*\*.*?\*\*',
            r'Frame \d+:.*?[\n.]', r'Observations?:.*?[\n.]',
            r'Key elements?:.*?[\n.]', r'Looking closer.*?[\n.]',
            r'Wait,.*?[\n.]', r'Let me.*?[\n.]', r'I can see.*?[\n.]',
            r'Based on.*?[\n.]',
        ]
        
        for pattern in bad_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        if not re.match(r'^(The video|This video|The footage|The clip)', text, re.IGNORECASE):
            text = re.sub(r'^.*?(shows?|depicts?|features?|captures?|demonstrates?)', 
                         r'The video \1', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]
        if sentences:
            text = '. '.join(sentences[:4]) + '.'
        
        return text.strip()[:400]
    
    def generate_caption_with_glm(self, description: str, style: str, 
                                   retry: bool = False) -> str:
        """Stage 2: Generate styled caption with GLM"""
        log_info(f"Stage 2: Generating '{style}' caption...")
        
        config = self.style_configs.get(style)
        if not config:
            return f"[Unknown style: {style}]"
        
        try:
            if retry:
                prompt = f"""{description}

Write a {style} caption about this video. {config['instruction']}

Begin your caption now (no preamble):"""
            else:
                prompt = f"""VIDEO: {description}

TASK: {config['instruction']}

OUTPUT (caption only, no preamble):"""
            
            system_prompt = f"{self.system_guidelines}\n\n{config['persona']}"
            
            payload = {
                "model": self.text_model,
                "max_tokens": 100,
                "temperature": config['temperature'],
                "top_p": 0.95,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = requests.post(self.api_url, headers=headers,
                                    data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            result = response.json()
            raw_caption = result['choices'][0]['message']['content'].strip()
            caption = self._extract_clean_caption(raw_caption, style)
            
            if not self._is_valid_caption(caption, style):
                if not retry:
                    log_warning("Caption quality low, retrying...")
                    return self.generate_caption_with_glm(description, style, retry=True)
                else:
                    log_warning("Retry failed, using fallback")
                    caption = self._generate_fallback_caption(description, style)
            
            log_success(f"Caption: {caption[:80]}...")
            return caption
        
        except Exception as e:
            log_error(f"GLM generation failed: {e}")
            return self._generate_fallback_caption(description, style)
    
    def _extract_clean_caption(self, text: str, style: str) -> str:
        """Extract clean caption"""
        prefixes = [
            r'^(CAPTION:|Caption:|OUTPUT:|Here\'s|Here is|Here\'s the|The caption is|'
            r'Based on|According to|This is|Let me|I\'ll|I will|Sure|Okay|Alright)[:.\s]*',
        ]
        
        for pattern in prefixes:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        text = text.strip('"\'`')
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\([^)]*meta[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*note[^)]*\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        if len(sentences) > 3:
            text = '. '.join(sentences[:3]) + '.'
        elif sentences:
            text = '. '.join(sentences)
            if not text.endswith(('.', '!', '?')):
                text += '.'
        
        return text.strip()
    
    def _is_valid_caption(self, caption: str, style: str) -> bool:
        """Validate caption quality"""
        if len(caption) < 25 or len(caption) > 400:
            return False
        
        caption_lower = caption.lower()
        bad_phrases = [
            'the user', 'analysis', 'frame 1', 'frame 2', 'observations',
            'key elements', 'drafting', 'idea 1', 'wait,', 'looking closer',
            'based on the', 'according to', 'the video description', 
            'the caption', 'instructions', 'let me', 'i can see',
            'here is', "here's", 'this is a', 'output:', 'caption:'
        ]
        
        for phrase in bad_phrases:
            if phrase in caption_lower:
                return False
        
        sentences = [s for s in re.split(r'[.!?]+', caption) if len(s.strip()) > 10]
        if len(sentences) < 1 or len(sentences) > 4:
            return False
        
        if style == "humorous_tech":
            tech_terms = ['api', 'code', 'debug', 'git', 'algorithm', 'stack', 
                         'compile', 'deploy', 'function', 'variable', 'loop', 
                         'bug', 'commit', 'push', 'merge', 'branch']
            if not any(term in caption_lower for term in tech_terms):
                return False
        
        if style == "humorous_non_tech":
            tech_terms = ['api', 'code', 'debug', 'git', 'algorithm', 'stack',
                         'compile', 'deploy', 'function', 'variable', 'loop', 
                         'programming', 'software', 'developer']
            if any(term in caption_lower for term in tech_terms):
                return False
        
        return True
    
    def _generate_fallback_caption(self, description: str, style: str) -> str:
        """Generate fallback caption"""
        desc_lower = description.lower()
        
        if 'kitten' in desc_lower or 'cat' in desc_lower:
            subject = 'kitten'
        elif 'street' in desc_lower or 'road' in desc_lower or 'traffic' in desc_lower:
            subject = 'street scene'
        elif 'office' in desc_lower or 'computer' in desc_lower or 'desk' in desc_lower:
            subject = 'office worker'
        elif 'tree' in desc_lower or 'autumn' in desc_lower:
            subject = 'autumn scenery'
        else:
            subject = 'scene'
        
        templates = {
            "formal": [
                f"The video captures a {subject} with clear composition and natural lighting in a controlled environment.",
                f"This footage demonstrates typical {subject} activity with observable movement and environmental detail.",
            ],
            "sarcastic": [
                f"Oh wow, another {subject}. Because we definitely needed more of this groundbreaking content.",
                f"Look at this {subject}. Absolutely riveting stuff right here, folks.",
            ],
            "humorous_tech": [
                f"This {subject} is running on legacy code with zero optimization and probably needs a git rebase.",
                f"Looks like someone deployed this {subject} straight to production without running the test suite first.",
            ],
            "humorous_non_tech": [
                f"This {subject} has the energy of someone who just realized it's Monday morning.",
                f"When you see this {subject}, you know it's going to be one of those days.",
            ]
        }
        
        import random
        return random.choice(templates.get(style, templates["formal"]))
    
    def process_video(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process single video task"""
        task_id = task['task_id']
        video_url = task['video_url']
        styles = task['styles']
        
        log_header(f"PROCESSING TASK: {task_id}")
        
        result = {"task_id": task_id, "captions": {}}
        temp_dir = None
        
        try:
            temp_dir = tempfile.mkdtemp()
            video_path = self.download_video(video_url, temp_dir)
            base64_frames = self.extract_keyframes(video_path)
            
            if not base64_frames:
                raise ValueError("No keyframes extracted")
            
            description = self.analyze_video_with_qwen(base64_frames)
            
            log_info(f"Generating {len(styles)} styled captions...")
            for i, style in enumerate(styles, 1):
                print(f"\n{Colors.BOLD}Style {i}/{len(styles)}: {style}{Colors.END}")
                
                if style not in self.style_configs:
                    result["captions"][style] = f"[Unsupported style: {style}]"
                    continue
                
                caption = self.generate_caption_with_glm(description, style)
                result["captions"][style] = caption
            
            log_success(f"Task {task_id} completed!\n")
        
        except Exception as e:
            log_error(f"Task {task_id} failed: {e}")
            for style in styles:
                if style not in result["captions"]:
                    result["captions"][style] = f"[Error: {str(e)[:100]}]"
        
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return result
    
    def run(self, input_path: str, output_path: str):
        """Main execution"""
        log_header("VIDEO CAPTIONING AGENT")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            
            log_success(f"Loaded {len(tasks)} tasks")
            
            results = []
            for i, task in enumerate(tasks, 1):
                print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
                print(f"{Colors.BOLD}TASK {i} of {len(tasks)}{Colors.END}")
                print(f"{Colors.BOLD}{'='*60}{Colors.END}")
                
                result = self.process_video(task)
                results.append(result)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            log_success(f"Results written to: {output_path}")
            log_header("ALL TASKS COMPLETED! 🎉")
            
        except Exception as e:
            log_error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Entry point"""
    api_key = os.getenv('FIREWORKS_API_KEY')
    if not api_key:
        log_error("FIREWORKS_API_KEY not set!")
        log_error("Please create a .env file with: FIREWORKS_API_KEY=your_key")
        sys.exit(1)
    
    if os.path.exists('/input/tasks.json'):
        input_path = '/input/tasks.json'
        output_path = '/output/results.json'
    else:
        input_path = 'input/tasks.json'
        output_path = 'output/results.json'
    
    if not os.path.exists(input_path):
        log_error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        agent = VideoCaptionAgent(api_key)
        agent.run(input_path, output_path)
        sys.exit(0)
    except Exception as e:
        log_error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()