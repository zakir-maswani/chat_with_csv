from datetime import datetime
import json
import logging
import os
import re
from typing import Dict, List, Optional, Any
from openai import OpenAI
from mistralai import Mistral
import base64
import traceback

from utils.logger_setup import setup_logging, log_and_print, log_only, print_only, Colors, setup_production_logging

class DocumentAnalyzer:
    """
    Analyzes PDF documents using Mistral OCR to extract structured information 
    including title, folder classification, summary, and tags.
    """
    
    def __init__(self, deepseek_client: OpenAI, mistral_client: Mistral, predefined_folders: Optional[List[str]] = None, used_tags: Optional[List[str]] = None,system_logger: logging.Logger = None):
        """
        Initialize the DocumentAnalyzer
        
        Args:
            deepseek_client: DeepSeek OpenAI-compatible client instance
            mistral_client: Mistral client instance
            predefined_folders: List of predefined folder names for classification
        """
        self.logger = system_logger if system_logger else setup_production_logging()
        log_and_print(self.logger, 'INFO', "Initializing DocumentAnalyzer", Colors.BLUE)
        self.deepseek_client = deepseek_client
        self.mistral_client = mistral_client
        self.predefined_folders = predefined_folders or [
            "Financial Documents",
            "Legal Documents", 
            "Medical Records",
            "Educational Materials",
            "Business Reports",
            "Personal Documents",
            "Technical Documentation",
            "Research Papers",
            "Contracts & Agreements",
            "Tax Documents",
            "Insurance Papers",
            "Real Estate",
            "Government Documents",
            "Correspondence",
            "Invoices & Receipts",
            "Policies & Procedures",
            "Meeting Notes",
            "Project Documentation",
            "Marketing Materials",
            "HR Documents",
            "Other"
        ]
        self.used_tags = used_tags or []
        log_only(self.logger, 'DEBUG', f"Initialized with {len(self.predefined_folders)} predefined folders")
    
    def extract_pdf_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from PDF using Mistral OCR
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Extracted text content
        """
        log_and_print(self.logger, 'INFO', f"Starting OCR extraction for: {os.path.basename(pdf_path)}", Colors.BLUE)
        try:
            # Encode PDF to base64
            print_only("🔄 Encoding PDF file...", Colors.CYAN)
            base64_pdf = self._encode_pdf(pdf_path)
            if not base64_pdf:
                log_and_print(self.logger, 'ERROR', "Failed to encode PDF to base64", Colors.RED)
                return ""
            
            log_only(self.logger, 'DEBUG', "PDF encoded successfully, calling Mistral OCR API")
            print_only("🔄 Processing with OCR...", Colors.CYAN)
            
            # Use Mistral OCR API
            ocr_response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=True
            )
            
            log_only(self.logger, 'DEBUG', "OCR API call successful, extracting text")
            # Extract text from OCR response
            extracted_text = ""
            if hasattr(ocr_response, 'text'):
                extracted_text = ocr_response.text
            elif hasattr(ocr_response, 'content'):
                extracted_text = ocr_response.content
            else:
                # Handle different response formats
                extracted_text = str(ocr_response)
            
            log_and_print(self.logger, 'INFO', f"OCR completed - extracted {len(extracted_text)} characters", Colors.GREEN)
            return extracted_text
            
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error in OCR extraction: {str(e)}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full traceback: {traceback.format_exc()}")
            return ""
    
    def _encode_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Encode PDF to base64
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Base64 encoded PDF string or None if error
        """
        try:
            with open(pdf_path, "rb") as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except FileNotFoundError:
            log_and_print(self.logger, 'ERROR', f"File not found: {pdf_path}", Colors.RED)
            return None
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error encoding PDF: {e}", Colors.RED)
            return None
    
    def analyze_document(self, document_text: str, document_name: Optional[str] = None, document_id: int = None, 
                        max_text_length: int = 8000) -> Dict[str, Any]:
        """
        Analyze a document and extract structured information
        
        Args:
            document_text: The full text content of the document
            document_name: Optional original document name/filename
            document_id: Optional document ID for tracking
            max_text_length: Maximum characters to analyze (to avoid token limits)
        
        Returns:
            Dictionary containing title, folder, summary, and tags
        """
        log_and_print(self.logger, 'INFO', f"Starting document analysis for: {document_name}", Colors.BLUE)
        log_only(self.logger, 'DEBUG', f"Input text length: {len(document_text)} characters")

        # Truncate text if too long
        analysis_text = document_text[:max_text_length] if len(document_text) > max_text_length else document_text
        
        # Create the analysis prompt
        folders_list = "\n".join([f"- {folder}" for folder in self.predefined_folders])
        used_tags_list = "\n".join([f"- {tag}" for tag in self.used_tags])

        system_prompt = f"""You are a document analysis expert. Your task is to analyze document content and extract structured information in JSON format.

For each document, provide:
1. TITLE: {document_name}
2. FOLDER: Choose the most appropriate folder from the predefined list
3. SUMMARY: Detailed Summary of the document content
4. TAG_CANDIDATES: 10 precise German tags with confidence scores (0.0-1.0), only include tags with confidence > 0.9


PREDEFINED FOLDERS:
{folders_list}

TAG CREATION GUIDELINES:
- Create 10 tags that are SPECIFIC to the document content and purpose
- Use standard German business/document terminology
- Focus on document type, subject matter, and key entities mentioned
- Avoid generic tags like "Dokument" or "Wichtig"
- Assign confidence scores between 0.0 and 1.0 for each tag
- ONLY include tags with confidence scores above 0.9 (90%)
- Examples of precise tags:
  * For invoices: "Rechnung", "Zahlungsaufforderung", "Kundennummer"
  * For contracts: "Vertrag", "Parteien", "Laufzeit"
  * For financial reports: "Jahresabschluss", "Bilanz", "Gewinn"
  * For medical documents: "Arztbrief", "Diagnose", "Behandlung"
  * For legal documents: "Anwaltsschreiben", "Klage", "Gericht"

IMPORTANT: 
- Always respond with valid JSON only
- Choose the most specific and appropriate folder
- Make the title descriptive but concise
- ALWAYS create tags in GERMAN LANGUAGE ONLY, regardless of document content language
- Create PRECISE and RELEVANT tags based on actual document content
- Only include tags with confidence scores above 0.9
- Summary should capture key information and purpose
- Use appropriate German terminology for the document type and content"""

        user_prompt = f"""Please analyze this document and provide structured information in JSON format:

DOCUMENT NAME: {document_name or "Unknown"}

DOCUMENT CONTENT:
{analysis_text}

TAG ANALYSIS INSTRUCTIONS:
1. Identify the document type (invoice, contract, report, etc.)
2. Identify key subjects, entities, or topics mentioned
3. Identify specific actions, dates, or amounts if relevant
4. Create 10 precise German tags with confidence scores (0.0-1.0) that capture the most important aspects
5. Only include tags with confidence scores above 0.9 (90%)

Provide your response as a JSON object with these exact keys: "title", "folder", "summary", "tag_candidates" (tag_candidates should be an array of objects with "tag" and "confidence" keys)."""

        try:
            print_only("🤖 Analyzing with AI...", Colors.CYAN)
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=8192,
                temperature=0.1,
                stream=False
            )
            
            log_only(self.logger, 'DEBUG', "Received AI response, parsing result")
            response_content = response.choices[0].message.content.strip()
            log_only(self.logger, 'DEBUG', f"Raw AI response: {response_content[:500]}...")
            
            # Try to parse the JSON response
            try:
                # Remove any markdown code blocks if present
                if response_content.startswith("```"):
                    response_content = re.sub(r'^```(?:json)?\n?', '', response_content)
                    response_content = re.sub(r'\n?```$', '', response_content)
                
                analysis_result = json.loads(response_content)
                
                # Validate required fields
                required_fields = ["title", "folder", "summary", "tag_candidates"]
                for field in required_fields:
                    if field not in analysis_result:
                        raise ValueError(f"Missing required field: {field}")
                
                # Ensure folder is from predefined list
                if analysis_result["folder"] not in self.predefined_folders:
                    analysis_result["folder"] = "Others"
                
                # Process tag candidates and select top 3 with confidence > 0.9
                tag_candidates = analysis_result.get("tag_candidates", [])
                log_only(self.logger, 'DEBUG', f"Tag candidates: {tag_candidates}")
                print_only("🏷️ Processing tags...", Colors.CYAN)
                
                if not isinstance(tag_candidates, list):
                    tag_candidates = []
                
                # Filter tags with confidence > 0.9 and sort by confidence
                high_confidence_tags = []
                for candidate in tag_candidates:
                    if isinstance(candidate, dict) and "tag" in candidate and "confidence" in candidate:
                        confidence = float(candidate["confidence"])
                        if confidence > 0.9:
                            high_confidence_tags.append({
                                "tag": candidate["tag"],
                                "confidence": confidence
                            })
                
                # Sort by confidence (highest first) and take top 3
                high_confidence_tags.sort(key=lambda x: x["confidence"], reverse=True)
                selected_tags = high_confidence_tags[:3]
                log_only(self.logger, 'DEBUG', f"Selected Tags: {selected_tags}")

                # LLM-based revalidation of selected tags
                def revalidate_tags_with_llm(document_text, selected_tags):
                    """
                    Use LLM to revalidate if the selected tags are truly relevant to the document text.
                    Returns a filtered list of tags that pass revalidation.
                    """
                    if not selected_tags:
                        return []
                    try:
                        print_only("🔍 Validating tags...", Colors.CYAN)
                        # Compose prompt for revalidation
                        tags_list = ', '.join([f'"{t["tag"]}"' for t in selected_tags])
                        prompt = f"""
Given the following document text and a list of selected tags, return only those tags that are truly relevant and well-supported by the document content. Only return tags that are clearly justified by the text. Respond with a JSON array of the valid tags (as strings, in German, no explanations).

Document Text:
{document_text[:4000]}

Selected Tags: [{tags_list}]
"""
                        # Use the same LLM client as for analysis
                        response = self.deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "You are a document tag validation expert. Only return valid tags as a JSON array."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=256,
                            temperature=0.1,
                            stream=False
                        )
                        content = response.choices[0].message.content.strip()
                        # Remove markdown if present
                        if content.startswith("```"):
                            content = re.sub(r'^```(?:json)?\n?', '', content)
                            content = re.sub(r'\n?```$', '', content)
                        valid_tags = json.loads(content)
                        # Only keep tags that are in the original selected_tags
                        valid_tags_set = set(valid_tags)
                        revalidated = [t for t in selected_tags if t["tag"] in valid_tags_set]
                        log_only(self.logger, 'DEBUG', f"Revalidated Tags: {revalidated}")
                        return revalidated
                    except Exception as e:
                        log_and_print(self.logger, 'WARNING', f"LLM revalidation failed: {e}", Colors.YELLOW)
                        return selected_tags  # fallback: return original

                # Call revalidation
                selected_tags = revalidate_tags_with_llm(document_text, selected_tags)
                
                # Extract just the tag names for final result
                final_tags = [tag_info["tag"] for tag_info in selected_tags]
                
                # Clean up the result - remove tag_candidates and only keep essential fields
                final_result = {
                    "title": analysis_result["title"],
                    "folder": analysis_result["folder"], 
                    "summary": analysis_result["summary"],
                    "tags": final_tags,
                    "analysis_metadata": {
                        "document_id": document_id,
                        "analyzed_at": datetime.now().isoformat(),
                        "original_filename": document_name,
                        "text_length": len(document_text),
                        "analyzed_length": len(analysis_text),
                        "truncated": len(document_text) > max_text_length
                    }
                }
                
                log_and_print(self.logger, 'INFO', "Document analysis completed successfully", Colors.GREEN)
                return final_result
                
            except json.JSONDecodeError as e:
                log_and_print(self.logger, 'ERROR', f"JSON parsing error: {str(e)}", Colors.RED)
                log_only(self.logger, 'DEBUG', f"Failed to parse: {response_content}")
                return self._create_fallback_analysis(document_text, document_name, document_id)
                
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Analysis error: {str(e)}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full traceback: {traceback.format_exc()}")
            return self._create_fallback_analysis(document_text, document_name, document_id)
    
    def _create_fallback_analysis(self, document_text: str, document_name: Optional[str], document_id: int = None) -> Dict[str, Any]:
        """
        Create a basic fallback analysis when AI analysis fails
        """
        log_and_print(self.logger, 'WARNING', "Using fallback analysis method", Colors.YELLOW)
        
        # Extract a simple title from document name or first line
        title = document_name or "Untitled Document"
        if not document_name and document_text:
            first_line = document_text.split('\n')[0][:80]
            if first_line.strip():
                title = first_line.strip()
        
        # Create basic summary
        summary = document_text[:200] + "..." if len(document_text) > 200 else document_text
        
        # Generate basic tags
        tags = []
        if document_name:
            # Extract words from filename
            filename_words = re.findall(r'\b\w{3,}\b', document_name.lower())
            tags.extend(filename_words[:3])
        
        # Add common words from text
        common_words = re.findall(r'\b\w{4,}\b', document_text.lower())
        word_freq = {}
        for word in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        tags.extend([word for word, freq in frequent_words if word not in tags])
        
        # Create tag candidates with confidence scores for fallback
        tag_candidates = []
        for i, tag in enumerate(tags[:8]):
            # Assign decreasing confidence scores for fallback tags
            confidence = max(0.95 - (i * 0.05), 0.85)
            tag_candidates.append({
                "tag": tag,
                "confidence": confidence
            })
        
        # Select top 3 tags for fallback
        selected_tags = tag_candidates[:3]
        final_tags = [tag_info["tag"] for tag_info in selected_tags]
        
        return {
            "title": title,
            "folder": "Others",
            "summary": summary,
            "tags": final_tags,
            "analysis_metadata": {
                "document_id": document_id,
                "analyzed_at": datetime.now().isoformat(),
                "original_filename": document_name,
                "text_length": len(document_text),
                "fallback_analysis": True,
                "error": "AI analysis failed, using fallback method"
            }
        }

    def analyze_pdf_file(self, pdf_path: str, document_id: int = None) -> Dict[str, Any]:
        """
        Analyze a PDF file using OCR and structured analysis
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document ID for tracking
        
        Returns:
            Analysis result dictionary
        """
        try:
            log_and_print(self.logger, 'INFO', f"Starting PDF analysis for: {os.path.basename(pdf_path)}", Colors.BLUE)
            document_text = self.extract_pdf_text_with_ocr(pdf_path)
            document_name = os.path.basename(pdf_path)
            
            if not document_text.strip():
                log_and_print(self.logger, 'WARNING', f"No text extracted from {document_name}", Colors.YELLOW)
                return self._create_fallback_analysis("", document_name, document_id)
            
            print_only("✅ OCR completed successfully", Colors.GREEN)
            log_only(self.logger, 'DEBUG', f"OCR successful - extracted {len(document_text)} characters")
            
            result = self.analyze_document(document_text, document_name, document_id)
            
            # Log results summary
            log_and_print(self.logger, 'INFO', f"PDF analysis complete for: {document_name}", Colors.GREEN)
            log_only(self.logger, 'DEBUG', f"Title: {result['title']}")
            log_only(self.logger, 'DEBUG', f"Folder: {result['folder']}")
            log_only(self.logger, 'DEBUG', f"Tags: {', '.join(result['tags'])}")
            log_only(self.logger, 'DEBUG', f"Summary length: {len(result['summary'])} chars")
            
            # Add OCR metadata to result
            result['analysis_metadata'].update({
                'ocr_processed': True,
                'ocr_text_length': len(document_text),
                'ocr_success': bool(document_text.strip())
            })
            
            return result
            
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error analyzing PDF file {pdf_path}: {e}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full traceback: {traceback.format_exc()}")
            return self._create_fallback_analysis("", os.path.basename(pdf_path), document_id)

    def analyze_text_document(self, document_text: str, document_name: Optional[str] = None, document_id: int = None, 
                            max_text_length: int = 8000) -> Dict[str, Any]:
        """
        Analyze a text document directly without OCR processing
        
        Args:
            document_text: The full text content of the document
            document_name: Optional original document name/filename  
            document_id: Optional document ID for tracking
            max_text_length: Maximum characters to analyze (to avoid token limits)
        
        Returns:
            Analysis result dictionary
        """
        try:
            log_and_print(self.logger, 'INFO', f"Starting text analysis for: {document_name}", Colors.BLUE)
            log_only(self.logger, 'DEBUG', f"Input text length: {len(document_text)} characters")
            
            if not document_text.strip():
                log_and_print(self.logger, 'WARNING', f"Empty text provided for document: {document_name}", Colors.YELLOW)
                return self._create_fallback_analysis("", document_name, document_id)
            
            # Analyze the text using DeepSeek
            result = self.analyze_document(document_text, document_name, document_id, max_text_length)
            
            # Log results summary
            log_and_print(self.logger, 'INFO', f"Text analysis complete for: {document_name}", Colors.GREEN)
            log_only(self.logger, 'DEBUG', f"Title: {result['title']}")
            log_only(self.logger, 'DEBUG', f"Folder: {result['folder']}")
            log_only(self.logger, 'DEBUG', f"Selected Tags: {', '.join(result['tags'])}")
            log_only(self.logger, 'DEBUG', f"Summary length: {len(result['summary'])} chars")
            
            # Add text processing metadata to result
            result['analysis_metadata'].update({
                'ocr_processed': False,
                'text_input_direct': True,
                'processing_method': 'direct_text_analysis'
            })
            
            return result
            
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Error analyzing text document {document_name}: {e}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full traceback: {traceback.format_exc()}")
            return self._create_fallback_analysis(document_text, document_name, document_id)


def analyze_pdf_document(pdf_path: str,system_logger:str, deepseek_api_key: str, mistral_api_key: str, new_folders: Optional[List[str]] = None, document_id: int = None, used_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a PDF document using Mistral OCR and DeepSeek analysis
    
    Args:
        pdf_path: Path to the PDF file
        deepseek_api_key: DeepSeek API key for text analysis
        mistral_api_key: Mistral API key for OCR
        new_folders: Optional list of custom folder names
        document_id: Optional document ID for tracking
    
    Returns:
        Analysis result dictionary
    """
    logger = system_logger if system_logger else setup_logging()
    log_and_print(logger, 'INFO', "Initializing PDF document analysis", Colors.BLUE)
    
    try:
        # Initialize clients
        deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        
        mistral_client = Mistral(api_key=mistral_api_key)
        
        # Initialize analyzer
        analyzer = DocumentAnalyzer(
            deepseek_client=deepseek_client,
            mistral_client=mistral_client,
            predefined_folders=new_folders,
            system_logger=logger,
        )
        
        # Process the PDF - OCR will happen first, then DeepSeek analysis
        return analyzer.analyze_pdf_file(pdf_path, document_id=document_id)
        
    except Exception as e:
        log_and_print(logger, 'ERROR', f"Failed to initialize PDF analysis: {str(e)}", Colors.RED)
        raise


def analyze_text_document(document_text: str, system_logger:str, document_name: Optional[str] = None, deepseek_api_key: str = None, 
                         new_folders: Optional[List[str]] = None, 
                         document_id: int = None, max_text_length: int = 8000) -> Dict[str, Any]:
    """
    Convenience function to analyze a text document directly using DeepSeek analysis
    
    Args:
        document_text: The full text content of the document
        document_name: Optional original document name/filename
        deepseek_api_key: DeepSeek API key for text analysis
        new_folders: Optional list of custom folder names
        document_id: Optional document ID for tracking
        max_text_length: Maximum characters to analyze (to avoid token limits)
    
    Returns:
        Analysis result dictionary
    """
    logger = system_logger if system_logger else setup_logging()
    log_and_print(logger, 'INFO', "Initializing text document analysis", Colors.BLUE)
    
    try:
        # Initialize DeepSeek client
        deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Initialize analyzer (no Mistral client needed for text analysis)
        analyzer = DocumentAnalyzer(
            deepseek_client=deepseek_client,
            mistral_client=None,  # Not needed for text analysis
            predefined_folders=new_folders,
        )
        
        # Process the text directly
        return analyzer.analyze_text_document(document_text, document_name, document_id, max_text_length)
        
    except Exception as e:
        log_and_print(logger, 'ERROR', f"Failed to initialize text analysis: {str(e)}", Colors.RED)
        raise


if __name__ == "__main__":
    logger = setup_production_logging()
    log_and_print(logger, 'INFO', "Starting document analysis script", Colors.BLUE)
    
    try:
        # Example usage for PDF analysis
        pdf_path = "2010-04-11 MAXUS_Brochure_eDELIVER7_HM_31082023_083.pdf"
        deepseek_key = ""
        mistral_key = ""
        
        # Example usage for text analysis
        sample_text = """
        Invoice Number: INV-2024-001
        Date: January 15, 2024
        
        Bill To:
        ABC Company
        123 Main Street
        City, State 12345
        
        Services Provided:
        - Consulting Services: $2,500.00
        - Project Management: $1,500.00
        
        Total Amount: $4,000.00
        Payment Due: February 15, 2024
        """
        
        print_only("="*50, Colors.MAGENTA)
        print_only("🚀 Document Analysis Demo", Colors.BOLD + Colors.WHITE)
        print_only("="*50, Colors.MAGENTA)
        
        log_and_print(logger, 'INFO', "Analyzing text document", Colors.BLUE)
        text_result = analyze_text_document(
            document_text=sample_text,
            document_name="Sample Invoice",
            deepseek_api_key=deepseek_key,
            new_folders=["Offers","Invoices", "Contracts","Other"],
            document_id=12345
        )
        
        if text_result:
            log_and_print(logger, 'INFO', "Text analysis completed, saving results", Colors.GREEN)
            with open("text_analysis_result.json", 'w', encoding='utf-8') as f:
                json.dump(text_result, f, indent=2, ensure_ascii=False)
            print_only("✅ Results saved to text_analysis_result.json", Colors.GREEN)
        
        log_and_print(logger, 'INFO', "Script execution completed successfully", Colors.GREEN)
        
    except Exception as e:
        log_and_print(logger, 'ERROR', f"Script execution failed: {str(e)}", Colors.RED)
        log_only(logger, 'DEBUG', f"Full traceback: {traceback.format_exc()}")