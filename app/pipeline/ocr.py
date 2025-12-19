import numpy as np
import easyocr
import re
from PIL import Image, ImageEnhance
from rapidfuzz.distance import Levenshtein
from app.config.settings import (
    GAMBLING_KEYWORDS,
    KEYWORD_WEIGHTS,
    SIMILARITY_THRESHOLD,
    MAGIC_NUMBER,
)

class GamblingOCR:
    def __init__(self):
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['id', 'en'], gpu=True)
        print("EasyOCR Ready")

    def preprocess_for_ocr(self, image: Image.Image):
        """
        Preprocessing: resize 2x + contrast enhancement
        Input: PIL Image
        Output: NumPy array for EasyOCR
        """
        # Resize image 2x using PIL (high-quality Lanczos resampling)
        width, height = image.size
        image_resized = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
        
        # Enhance contrast using PIL (equivalent to CLAHE but simpler)
        enhancer = ImageEnhance.Contrast(image_resized)
        image_enhanced = enhancer.enhance(1.5)  # 1.5x contrast boost
        
        # Convert PIL to NumPy array for EasyOCR (RGB format)
        img_array = np.array(image_enhanced)
        
        return img_array
    
    def normalize(self, text):
        """Normalize text: lowercase, replace numbers, remove special chars"""
        text = text.lower()
        text = (
            text.replace("4", "a").replace("3", "e")
                .replace("1", "i").replace("0", "o")
                .replace("5", "s")
        )
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return text.split()
    
    def ngrams(self, tokens, n=2):
        """Generate n-grams from tokens"""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def levenshtein_similarity(self, a, b):
        """Calculate Levenshtein similarity (0.0 - 1.0)"""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        distance = Levenshtein.distance(a, b)
        max_len = max(len(a), len(b))
        return 1 - (distance / max_len)
    
    def score_text(self, tokens):
        """Score text based on gambling keyword matching"""
        score = 0.0
        
        # Unigrams
        for t in tokens:
            for kw in GAMBLING_KEYWORDS:
                if " " in kw:
                    continue
                sim = self.levenshtein_similarity(t, kw)
                if sim >= SIMILARITY_THRESHOLD:
                    w = KEYWORD_WEIGHTS.get(kw, 1)
                    score += sim * w
        
        # Bigrams & trigrams
        bigrams = self.ngrams(tokens, 2)
        trigrams = self.ngrams(tokens, 3)
        for phrase in bigrams + trigrams:
            for kw in GAMBLING_KEYWORDS:
                if " " not in kw:
                    continue
                sim = self.levenshtein_similarity(phrase, kw)
                if sim >= SIMILARITY_THRESHOLD:
                    w = KEYWORD_WEIGHTS.get(kw, 2)
                    score += sim * w
        
        return score
    
    def classify_gambling_ocr(self, image: Image.Image):
        """
        OCR Heuristic: Read full image, match keywords, return prob + text
        Input: PIL Image (consistent with ViT and RT-DETR)
        Output:
          - prob_gambling (0.0 - 1.0)
          - label_ocr (gambling / non_gambling)
          - ocr_text (raw text from image)
        """
        # Preprocess PIL Image and convert to NumPy for EasyOCR
        img_array = self.preprocess_for_ocr(image)
        
        # Read text with EasyOCR (accepts NumPy array)
        texts = self.reader.readtext(img_array, detail=0)
        raw_text = " ".join(texts)
        
        # Normalize and tokenize
        norm = self.normalize(raw_text)
        tokens = self.tokenize(norm)
        
        # Score based on keyword matching
        score = self.score_text(tokens)
        
        # Normalize score to 0-1
        normalized_score = score / MAGIC_NUMBER
        if normalized_score > 1.0:
            normalized_score = 1.0
        
        prob_gambling = float(normalized_score)
        label_ocr = "gambling" if prob_gambling >= 0.5 else "non_gambling"
        
        return prob_gambling, label_ocr, raw_text
