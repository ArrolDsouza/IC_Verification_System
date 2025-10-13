import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
import pandas as pd
import re
import os
import threading
from PIL import Image, ImageTk
from typing import Dict, List, Optional, Tuple
import csv

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✓ EasyOCR available")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("✗ EasyOCR not installed")

class ICImageProcessor:
    def __init__(self):
        self.target_size = (1600, 1200)
    
    def get_all_preprocessed_versions(self, image_path: str) -> List[Tuple[str, np.ndarray]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        image = cv2.resize(image, self.target_size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        versions = []
        versions.append(("original_gray", gray.copy()))
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        versions.append(("clahe", clahe_enhanced))
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        versions.append(("otsu_binary", binary))
        
        _, inv_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        versions.append(("inverted_binary", inv_binary))
        
        adaptive_gauss = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        versions.append(("adaptive_gaussian", adaptive_gauss))
        
        adaptive_mean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        versions.append(("adaptive_mean", adaptive_mean))
        
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        versions.append(("morphological", morph))
        
        contrast_enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        versions.append(("high_contrast", contrast_enhanced))
        
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        versions.append(("denoised", denoised))
        
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)
        versions.append(("sharpened", sharpened))
        
        print(f"\nGenerated {len(versions)} preprocessed image versions")
        return versions

class ICTextExtractor:
    def __init__(self):
        self.processor = ICImageProcessor()
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE:
            try:
                print("Initializing EasyOCR (this may take a moment)...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("✓ EasyOCR initialized successfully")
            except Exception as e:
                print(f"✗ Error initializing EasyOCR: {e}")
    
    def extract_with_easyocr(self, image: np.ndarray, strategy_name: str) -> List[str]:
        if not self.easyocr_reader:
            return []
        
        try:
            results = self.easyocr_reader.readtext(image, paragraph=False)
            
            texts = []
            for bbox, text, confidence in results:
                if confidence > 0.1 and len(text.strip()) > 0:
                    texts.append(text.strip())
                    print(f"  EasyOCR [{strategy_name}]: '{text.strip()}' (conf: {confidence:.2f})")
            
            return texts
            
        except Exception as e:
            print(f"  Error in EasyOCR [{strategy_name}]: {e}")
            return []
    
    def extract_text_from_image(self, image_path: str) -> Tuple[List[str], Dict]:
        try:
            print(f"\n{'='*60}")
            print(f"Extracting text from: {os.path.basename(image_path)}")
            print(f"{'='*60}")
            
            preprocessed_versions = self.processor.get_all_preprocessed_versions(image_path)
            all_extracted_texts = []
            
            for strategy_name, processed_image in preprocessed_versions:
                print(f"\nTrying strategy: {strategy_name}")
                if EASYOCR_AVAILABLE:
                    easyocr_texts = self.extract_with_easyocr(processed_image, strategy_name)
                    all_extracted_texts.extend(easyocr_texts)
            
            cleaned_texts = []
            seen_normalized = set()
            
            for text in all_extracted_texts:
                cleaned = self.clean_text(text)
                if cleaned and len(cleaned) >= 2:
                    normalized = cleaned.upper().replace(' ', '').replace('-', '')
                    if normalized not in seen_normalized:
                        cleaned_texts.append(cleaned)
                        seen_normalized.add(normalized)
                    
                    if 'Z' in cleaned or 'O' in cleaned:
                        corrected = cleaned.replace('Z', '7').replace('O', '0')
                        corrected_norm = corrected.upper().replace(' ', '').replace('-', '')
                        if corrected_norm not in seen_normalized:
                            cleaned_texts.append(corrected)
                            seen_normalized.add(corrected_norm)
            
            print(f"\n{'='*60}")
            print(f"FINAL EXTRACTED TEXTS ({len(cleaned_texts)} unique):")
            for i, text in enumerate(cleaned_texts, 1):
                print(f"  {i}. '{text}'")
            print(f"{'='*60}\n")
            
            original_gray = preprocessed_versions[0][1]
            quality_metrics = self.calculate_quality_metrics(original_gray)
            
            return cleaned_texts, quality_metrics
            
        except Exception as e:
            print(f"✗ Error extracting text: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], {}
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        
        special_chars = len(re.findall(r'[^A-Za-z0-9\s\-/]', text))
        if special_chars > len(text) * 0.4:
            return ""
        
        if not re.search(r'[A-Za-z0-9]', text):
            return ""
        
        text = text.upper()
        text = self.correct_ocr_errors(text)
        return text
    
    def correct_ocr_errors(self, text: str) -> str:
        corrections = {
            'Z': '7',
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8',
        }
        corrected_versions = [text]
        if 'SN' in text and 'Z' in text:
            corrected_versions.append(text.replace('Z', '7'))
        if 'O' in text and any(char.isdigit() for char in text):
            corrected_versions.append(text.replace('O', '0'))
        if any(char.isdigit() for char in text):
            corrected_versions.sort(key=lambda x: sum(c.isdigit() for c in x), reverse=True)
        return corrected_versions[0]
    
    def calculate_quality_metrics(self, image: np.ndarray) -> Dict:
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        contrast = np.std(image)
        brightness = np.mean(image)
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'sharpness': laplacian_var,
            'contrast': contrast,
            'brightness': brightness,
            'edge_density': edge_density
        }

class ICVerifier:
    def __init__(self, csv_path: str = "IC_DATA.csv"):
        self.csv_path = csv_path
        self.ic_database = self.load_database()
        self.all_part_numbers = set()
        self.load_all_part_numbers()
    
    def load_database(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.csv_path):
                print(f"✗ Warning: {self.csv_path} not found")
                return pd.DataFrame()
            
            df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded {len(df)} IC records from database")
            
            if len(df) > 0:
                print(f"  Sample part numbers: {list(df['part_number'].head(3))}")
            
            return df
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            return pd.DataFrame()
    
    def load_all_part_numbers(self):
        if self.ic_database.empty:
            return
        
        for col in ['part_number', 'part_number_text', 'manufacturer_logo_text']:
            if col in self.ic_database.columns:
                for val in self.ic_database[col].dropna():
                    self.all_part_numbers.add(str(val).strip().upper())
    
    def normalize_text(self, text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', text.upper())
    
    def find_matching_ic(self, extracted_texts: List[str]) -> Optional[Dict]:
        if self.ic_database.empty or not extracted_texts:
            return None
        
        print(f"\nSearching database for matches...")
        print(f"Extracted texts to match: {extracted_texts}")
        
        best_match = None
        best_score = 0
        
        for idx, row in self.ic_database.iterrows():
            part_number = str(row.get('part_number', '')).strip()
            manufacturer = str(row.get('manufacturer', '')).strip()
            part_text = str(row.get('part_number_text', '')).strip()
            mfg_text = str(row.get('manufacturer_logo_text', '')).strip()
            
            db_texts = [part_number, part_text, mfg_text]
            db_texts = [t for t in db_texts if t and t != 'nan']
            
            for extracted in extracted_texts:
                extracted_norm = self.normalize_text(extracted)
                
                if len(extracted_norm) < 3:
                    continue
                
                for db_text in db_texts:
                    db_norm = self.normalize_text(db_text)
                    
                    if extracted_norm == db_norm:
                        print(f"✓ EXACT MATCH: '{extracted}' == '{db_text}'")
                        return {
                            'id': row.get('id'),
                            'manufacturer': manufacturer,
                            'part_number': part_number,
                            'description': row.get('description', 'N/A'),
                            'match_score': 1.0,
                            'matched_text': extracted,
                            'match_type': 'exact'
                        }
                    
                    similarity = self.calculate_similarity(extracted_norm, db_norm)
                    
                    if similarity > best_score and similarity >= 0.7:
                        best_score = similarity
                        best_match = {
                            'id': row.get('id'),
                            'manufacturer': manufacturer,
                            'part_number': part_number,
                            'description': row.get('description', 'N/A'),
                            'match_score': similarity,
                            'matched_text': extracted,
                            'match_type': 'fuzzy'
                        }
                        print(f"  Fuzzy match: '{extracted}' ~ '{db_text}' (similarity: {similarity:.2f})")
                    
                    if extracted_norm in db_norm or db_norm in extracted_norm:
                        score = len(min(extracted_norm, db_norm, key=len)) / len(max(extracted_norm, db_norm, key=len))
                        if score > best_score:
                            best_score = score
                            best_match = {
                                'id': row.get('id'),
                                'manufacturer': manufacturer,
                                'part_number': part_number,
                                'description': row.get('description', 'N/A'),
                                'match_score': score,
                                'matched_text': extracted,
                                'match_type': 'partial'
                            }
                            print(f"  Partial match: '{extracted}' ~ '{db_text}' (score: {score:.2f})")
        
        if best_match and best_score >= 0.7:
            print(f"✓ Best match found: {best_match['part_number']} (score: {best_score:.2f})")
            return best_match
        
        print("✗ No database match found")
        return None
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        if str1 == str2:
            return 1.0
        
        len1, len2 = len(str1), len(str2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        matches = 0
        for i, char in enumerate(str1):
            if i < len2 and str1[i] == str2[i]:
                matches += 1
        
        char_overlap = len(set(str1) & set(str2))
        positional_score = matches / max(len1, len2)
        overlap_score = char_overlap / len(set(str1) | set(str2))
        similarity = (positional_score * 0.7) + (overlap_score * 0.3)
        
        return similarity

class FakeICDetector:
    def detect_fake_indicators(self, extracted_texts: List[str], 
                               quality_metrics: Dict, 
                               match_result: Optional[Dict]) -> Dict:
        fake_indicators = []
        confidence_penalty = 0
        
        if len(extracted_texts) == 0:
            fake_indicators.append("NO TEXT DETECTED - Cannot verify IC")
            confidence_penalty += 0.6
        
        if not match_result:
            fake_indicators.append("No database match found")
            confidence_penalty += 0.5
        elif match_result['match_score'] < 0.7:
            fake_indicators.append(f"Weak database match ({match_result['match_score']:.1%})")
            confidence_penalty += 0.3
        
        if quality_metrics.get('sharpness', 0) < 100:
            if not match_result or match_result['match_score'] < 0.9:
                fake_indicators.append("Low image sharpness (may affect accuracy)")
                confidence_penalty += 0.05
            else:
                fake_indicators.append("Low image sharpness (but strong database match found)")
        
        if match_result and match_result['match_score'] >= 0.95:
            is_likely_fake = False
            confidence_penalty = min(0.1, confidence_penalty)
        elif match_result and match_result['match_score'] >= 0.8:
            is_likely_fake = confidence_penalty >= 0.4
        else:
            is_likely_fake = len(fake_indicators) >= 2 or confidence_penalty >= 0.5
        
        return {
            'fake_indicators': fake_indicators,
            'confidence_penalty': min(1.0, confidence_penalty),
            'is_likely_fake': is_likely_fake
        }

class ICDetectionSystem:
    def __init__(self, csv_path: str = "IC_DATA.csv"):
        self.extractor = ICTextExtractor()
        self.verifier = ICVerifier(csv_path)
        self.fake_detector = FakeICDetector()
    
    def analyze_image(self, image_path: str) -> Dict:
        try:
            print(f"\n{'#'*60}")
            print(f"=== Analyzing image: {os.path.basename(image_path)} ===")
            print(f"{'#'*60}")
            
            extracted_texts, quality_metrics = self.extractor.extract_text_from_image(image_path)
            
            print(f"\nExtracted {len(extracted_texts)} unique text elements:")
            for text in extracted_texts:
                print(f"  - {text}")
            
            match_result = self.verifier.find_matching_ic(extracted_texts)
            
            fake_analysis = self.fake_detector.detect_fake_indicators(
                extracted_texts, quality_metrics, match_result
            )
            
            if match_result and match_result['match_score'] >= 0.6:
                is_real = not fake_analysis['is_likely_fake']
                confidence = match_result['match_score'] - fake_analysis['confidence_penalty']
                prediction = 'REAL' if is_real else 'FAKE'
            else:
                is_real = False
                confidence = 0.0
                prediction = 'FAKE'
            
            print(f"\nFinal prediction: {prediction} (confidence: {confidence:.1%})")
            
            return {
                'image_path': image_path,
                'extracted_texts': extracted_texts,
                'quality_metrics': quality_metrics,
                'match_result': match_result,
                'fake_analysis': fake_analysis,
                'is_real': is_real,
                'confidence': max(0.0, min(1.0, confidence)),
                'prediction': prediction
            }
            
        except Exception as e:
            print(f"✗ Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'image_path': image_path,
                'extracted_texts': [],
                'quality_metrics': {},
                'match_result': None,
                'fake_analysis': {
                    'fake_indicators': [f"Error: {str(e)}"],
                    'confidence_penalty': 1.0,
                    'is_likely_fake': True
                },
                'is_real': False,
                'confidence': 0.0,
                'prediction': 'ERROR'
            }

class ICDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IC Verification System")
        self.root.geometry("1100x750")
        self.root.configure(bg='#f0f0f0')
        
        self.detection_system = ICDetectionSystem()
        self.current_image_path = None
        self.current_image = None
        self.analysis_result = None
        
        self.create_widgets()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="IC Varification System ", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
        
        left_panel = ttk.LabelFrame(main_frame, text="Image Analysis", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_panel.grid_rowconfigure(1, weight=1)
        
        controls = ttk.Frame(left_panel)
        controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(controls, text="Select Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        self.analyze_button = ttk.Button(controls, text="Analyze", command=self.analyze_image, state='disabled')
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        
        self.image_label = ttk.Label(left_panel, text="No image selected", background='white')
        self.image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.status_label = ttk.Label(left_panel, text="Ready", foreground='green')
        self.status_label.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        right_panel = ttk.LabelFrame(main_frame, text="Results", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.grid_rowconfigure(1, weight=1)
        
        self.result_label = ttk.Label(right_panel, text="No analysis", font=('Arial', 12, 'bold'))
        self.result_label.grid(row=0, column=0, pady=(0, 10))
        
        self.results_text = scrolledtext.ScrolledText(right_panel, height=25, width=55, wrap=tk.WORD)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def select_image(self):
        filename = filedialog.askopenfilename(
            title="Select IC Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if filename:
            self.current_image_path = filename
            self.load_image()
            self.analyze_button.config(state='normal')
            self.status_label.config(text=f"Loaded: {os.path.basename(filename)}", foreground='blue')
    
    def load_image(self):
        try:
            image = Image.open(self.current_image_path)
            image.thumbnail((450, 350), Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.current_image, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {str(e)}")
    
    def analyze_image(self):
        if not self.current_image_path:
            return
        
        self.progress.start()
        self.status_label.config(text="Analyzing (check console for details)...", foreground='orange')
        self.analyze_button.config(state='disabled')
        
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        result = self.detection_system.analyze_image(self.current_image_path)
        self.analysis_result = result
        self.root.after(0, self.update_results)
    
    def update_results(self):
        self.progress.stop()
        self.analyze_button.config(state='normal')
        
        if not self.analysis_result:
            self.status_label.config(text="Failed", foreground='red')
            return
        
        result = self.analysis_result
        prediction = result['prediction']
        confidence = result['confidence']
        
        if prediction == 'REAL':
            self.result_label.config(text=f"✓ REAL IC ({confidence:.0%})", foreground='green')
        else:
            self.result_label.config(text=f"✗ FAKE IC ({confidence:.0%})", foreground='red')
        
        self.results_text.delete(1.0, tk.END)
        details = [
            f"File: {os.path.basename(result['image_path'])}\n",
            f"Prediction: {prediction}",
            f"Confidence: {confidence:.1%}\n",
            "="*50,
            "\nExtracted Texts:"
        ]
        
        if result['extracted_texts']:
            for i, txt in enumerate(result['extracted_texts'], 1):
                details.append(f"  {i}. {txt}")
        else:
            details.append("  ⚠ NO TEXT DETECTED")
            details.append("  Check console for OCR details")
        
        details.append("")
        
        if result['match_result']:
            m = result['match_result']
            details.extend([
                "Database Match:",
                f"  Manufacturer: {m['manufacturer']}",
                f"  Part Number: {m['part_number']}",
                f"  Match Score: {m['match_score']:.1%}",
                f"  Type: {m.get('match_type', 'N/A')}\n"
            ])
        
        if result['fake_analysis']['fake_indicators']:
            details.append("Fake Indicators:")
            for ind in result['fake_analysis']['fake_indicators']:
                details.append(f"  • {ind}")
        
        self.results_text.insert(1.0, "\n".join(details))
        self.status_label.config(text="Complete - Check console for details", foreground='green')

def main():
    print("\n" + "="*60)
    print("IC Detection System - Ultra Robust OCR")
    print("="*60)
    print(f"EasyOCR: {'✓ Available' if EASYOCR_AVAILABLE else '✗ Not installed'}")
    print("="*60 + "\n")
    
    if not EASYOCR_AVAILABLE:
        print("WARNING: No OCR libraries installed!")
        print("Install EasyOCR: pip install easyocr\n")
    
    root = tk.Tk()
    app = ICDetectorGUI(root)
    
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
