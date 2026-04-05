import os
import base64
import time
import json
import shutil
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from mistralai import Mistral
from pydantic import BaseModel, Field
from enum import Enum
import PyPDF2
import logging
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import io
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Enhanced Pydantic models for structured data extraction ---
class PageType(str, Enum):
    TEXT = "text"
    CHART = "chart"
    TABLE = "table"
    DIAGRAM = "diagram"
    MIXED = "mixed"
    COVER = "cover"
    INDEX = "index"
    BLANK_ARTIFACT = "blank_artifact"

class FolderType(str, Enum):
    CONTRACTS = "contracts"
    OFFERS = "offers"  
    INVOICES = "invoices"
    SALES_REPORTS = "sales_reports"
    OTHERS = "others"

class PageMetadata(BaseModel):
    page_number: int = Field(..., description="The page number in the document")
    file_name: str = Field(..., description="Name of the source PDF file")
    folder_type: FolderType = Field(default=FolderType.OTHERS, description="Type of document folder")
    page_type: PageType = Field(..., description="Type of content on this page")
    title: Optional[str] = Field(None, description="Title or main heading of the page")
    summary: str = Field(..., description="Brief summary of what this page contains")
    key_topics: List[str] = Field(default=[], description="Main topics or concepts covered")
    
    # Enhanced metadata fields
    important_facts: List[str] = Field(default=[], description="Key facts or important information")
    numerical_data: List[Dict[str, Any]] = Field(default=[], description="Important numbers, dates, figures")
    action_items: List[str] = Field(default=[], description="Action items or tasks mentioned")
    people_mentioned: List[str] = Field(default=[], description="Names of people mentioned")
    organizations: List[str] = Field(default=[], description="Organizations or companies mentioned")
    dates_mentioned: List[str] = Field(default=[], description="Important dates mentioned")
    
    # Visual elements
    has_tables: bool = Field(default=False, description="Whether the page contains tables")
    has_charts: bool = Field(default=False, description="Whether the page contains charts or graphs")
    has_images: bool = Field(default=False, description="Whether the page contains images")
    
    # Quality metrics
    confidence_score: float = Field(default=0.0, description="Confidence in the analysis (0-1)")
    readability_score: float = Field(default=0.0, description="How readable/clear the content is (0-1)")
    
    # Folder-specific fields
    folder_specific_data: Dict[str, Any] = Field(default={}, description="Folder-specific extracted information")

class ProcessedPage(BaseModel):
    metadata: PageMetadata
    markdown_content: str = Field(..., description="The page content in markdown format")
    processing_time: float = Field(..., description="Time taken to process this page in seconds")

# --- Folder-Specific Prompt Generator ---
class FolderPromptGenerator:
    @staticmethod
    def get_folder_type_by_id(folder_id: int) -> FolderType:
        """Determine folder type from folder ID"""
        folder_id_mappings = {
            1: FolderType.SALES_REPORTS,
            2: FolderType.CONTRACTS,
            3: FolderType.OFFERS,
            5: FolderType.INVOICES,
            6: FolderType.OTHERS
        }
        return folder_id_mappings.get(folder_id, FolderType.OTHERS)
    
    @staticmethod
    def get_folder_type(folder: str) -> FolderType:
        """Determine folder type from folder name or ID"""
        folder_lower = folder.lower().strip()
        
        # Direct folder name mapping
        folder_mappings = {
            'contracts': FolderType.CONTRACTS,
            'offers': FolderType.OFFERS,
            'invoices': FolderType.INVOICES,
            'sales reports': FolderType.SALES_REPORTS,
            'others': FolderType.OTHERS
        }
        
        # Check for exact match first
        if folder_lower in folder_mappings:
            return folder_mappings[folder_lower]
        
        # Check for partial matches
        if any(keyword in folder_lower for keyword in ['contract', 'agreement', 'legal']):
            return FolderType.CONTRACTS
        elif any(keyword in folder_lower for keyword in ['offer', 'proposal', 'quote', 'bid']):
            return FolderType.OFFERS
        elif any(keyword in folder_lower for keyword in ['invoice', 'bill', 'receipt', 'payment']):
            return FolderType.INVOICES
        elif any(keyword in folder_lower for keyword in ['sales', 'report', 'performance', 'revenue']):
            return FolderType.SALES_REPORTS
        else:
            return FolderType.OTHERS

    @staticmethod
    def generate_prompt(folder_type: FolderType, page_number: int, file_name: str) -> str:
        """Generate folder-specific analysis prompt with enhanced important data extraction"""
        
        base_instructions = f"""
        Analyze this document page image carefully and provide detailed information. This is page {page_number} from the file "{file_name}".

        CRITICAL INSTRUCTIONS FOR DATA EXTRACTION:
        1. Extract ALL important data elements - be comprehensive but smart
        2. For repetitive content, mention once and note repetition 
        3. Focus on UNIQUE, MEANINGFUL content only
        4. Extract specific values, names, dates, amounts with precision
        5. Identify relationships between data elements
        6. Note data patterns and structures (especially in tables/forms)
        7. Extract both explicit information AND implied/calculated information
        8. For numerical data, include context (what the number represents)
        9. For dates, include context (what event/deadline the date relates to)
        10. For names, include their role/context when mentioned
        """

        folder_specific_instructions = {
            FolderType.CONTRACTS: """
            CONTRACT DOCUMENT - EXTRACT THESE KEY DATA ELEMENTS:
            
            PARTIES & ENTITIES:
            - All contracting parties (full legal names, addresses, contact info)
            - Authorized signatories and their titles
            - Witnesses, notaries, legal representatives
            - Parent companies, subsidiaries, affiliated entities
            
            FINANCIAL DATA:
            - Contract value/amount (total and breakdown)
            - Payment schedules and amounts
            - Penalty amounts and calculation methods
            - Deposit amounts and terms
            - Rate information (hourly, monthly, per unit)
            - Currency and exchange rate information
            - Tax implications and amounts
            
            TEMPORAL DATA:
            - Contract start and end dates
            - Signature dates and execution dates
            - Payment due dates and milestones
            - Renewal dates and notification periods
            - Deadline dates for deliverables
            - Notice periods for termination
            
            LEGAL & COMPLIANCE DATA:
            - Governing law and jurisdiction
            - Regulatory requirements and standards
            - Insurance requirements and amounts
            - Intellectual property clauses
            - Confidentiality terms and duration
            - Liability limits and exclusions
            
            PERFORMANCE DATA:
            - Deliverables and specifications
            - Service level agreements (SLAs)
            - Quality standards and metrics
            - Performance benchmarks and KPIs
            - Acceptance criteria and testing procedures
            """,
            
            FolderType.OFFERS: """
            OFFER/PROPOSAL DOCUMENT - EXTRACT THESE KEY DATA ELEMENTS:
            
            COMMERCIAL DATA:
            - Total offer value and currency
            - Itemized pricing with quantities and unit prices
            - Discount percentages and amounts
            - Tax rates and amounts
            - Payment terms and methods
            - Validity period and expiration dates
            
            TECHNICAL SPECIFICATIONS:
            - Product/service descriptions and specifications
            - Technical requirements and standards
            - Capacity, performance, or quality metrics
            - Delivery specifications and packaging
            - Installation or implementation requirements
            
            DELIVERY & TIMELINE DATA:
            - Delivery dates and schedules
            - Implementation timelines and milestones
            - Lead times and production schedules
            - Shipping methods and terms
            - Installation or setup timeframes
            
            TERMS & CONDITIONS:
            - Warranty periods and coverage
            - Support and maintenance terms
            - Return or cancellation policies
            - Terms of acceptance
            - Liability and risk allocation
            
            COMPETITIVE DATA:
            - Unique value propositions
            - Comparative advantages mentioned
            - Market positioning statements
            - Competitive differentiation points
            """,
            
            FolderType.INVOICES: """
            INVOICE DOCUMENT - EXTRACT THESE KEY DATA ELEMENTS:
            
            INVOICE IDENTIFICATION:
            - Invoice number and reference codes
            - Purchase order numbers
            - Job or project numbers
            - Customer reference numbers
            
            FINANCIAL DATA:
            - Line item descriptions, quantities, unit prices
            - Subtotal amounts before taxes
            - Tax rates, tax amounts, tax registration numbers
            - Total amount due
            - Previously paid amounts or credits
            - Outstanding balance
            - Currency and exchange rates
            
            PAYMENT DATA:
            - Payment terms (net 30, etc.)
            - Due dates and late payment penalties
            - Payment methods accepted
            - Bank details for transfers
            - Early payment discounts
            
            PARTY INFORMATION:
            - Seller/vendor details (name, address, contact info)
            - Buyer/customer details (billing and shipping addresses)
            - Tax identification numbers
            - Business registration numbers
            
            SHIPPING & DELIVERY:
            - Shipping dates and methods
            - Delivery addresses and contacts
            - Tracking numbers and carriers
            - Freight charges and terms
            """,
            
            FolderType.SALES_REPORTS: """
            SALES REPORT - EXTRACT THESE KEY DATA ELEMENTS:
            
            PERFORMANCE METRICS:
            - Revenue figures (actual vs. target)
            - Sales volume (units, transactions)
            - Growth percentages (YoY, QoQ, MoM)
            - Profit margins and profitability ratios
            - Market share percentages
            - Conversion rates and pipeline metrics
            
            TEMPORAL DATA:
            - Reporting period (start and end dates)
            - Comparison periods
            - Seasonal trends and patterns
            - Forecast periods and projections
            
            SEGMENTATION DATA:
            - Sales by product line or category
            - Sales by geographic region
            - Sales by customer segment
            - Sales by channel or method
            - Performance by sales representative/team
            
            CUSTOMER DATA:
            - New customer acquisitions
            - Customer retention rates
            - Average customer value
            - Customer lifetime value
            - Lost customers and churn rates
            
            MARKET INTELLIGENCE:
            - Competitive positioning
            - Market trends and opportunities
            - Pricing analysis and elasticity
            - Demand forecasting data
            - Market penetration rates
            """,
            
            FolderType.OTHERS: """
            GENERAL DOCUMENT - EXTRACT THESE KEY DATA ELEMENTS:
            
            IDENTIFICATION DATA:
            - Document type and purpose
            - Document number or identifier
            - Version or revision information
            - Classification or category
            
            STAKEHOLDER DATA:
            - All mentioned individuals and their roles
            - Organizations and their relationships
            - Contact information and addresses
            - Authority levels and responsibilities
            
            FINANCIAL DATA:
            - Any monetary amounts and their context
            - Budget figures and allocations
            - Cost estimates and actual expenses
            - Revenue or income figures
            
            OPERATIONAL DATA:
            - Process descriptions and workflows
            - Procedures and protocols
            - Standards and requirements
            - Metrics and measurements
            
            TEMPORAL DATA:
            - All dates and their significance
            - Deadlines and milestones
            - Schedules and timelines
            - Historical references
            """
        }

        folder_instruction = folder_specific_instructions.get(folder_type, folder_specific_instructions[FolderType.OTHERS])

        return f"""
        {base_instructions}
        
        {folder_instruction}

        Please provide a comprehensive analysis in the following format:

        **PAGE ANALYSIS:**
        
        **TITLE:** [Extract the main title/heading if visible, or "No clear title" if none]
        
        **PAGE TYPE:** [Classify as: text, chart, table, diagram, mixed, cover, index, or blank_artifact]
        
        **SUMMARY:** [Provide a 2-3 sentence summary focusing on the main purpose and content]
        
        **KEY TOPICS:** [List 3-5 main topics or concepts, separated by commas]
        
        **IMPORTANT FACTS:** [List key facts, findings, or critical information - separate each with ||]
        
        **NUMERICAL DATA:** [Extract important numbers, dates, amounts, percentages - format as "label: value" separated by ||]
        
        **ACTION ITEMS:** [List any tasks, requirements, or action items mentioned - separate with ||]
        
        **PEOPLE MENTIONED:** [List names of people mentioned with their roles when possible - separate with ||]
        
        **ORGANIZATIONS:** [List companies, institutions, organizations mentioned - separate with ||]
        
        **DATES MENTIONED:** [List important dates in format "event: date" - separate with ||]
        
        **VISUAL ELEMENTS:**
        - Has tables: [Yes/No]
        - Has charts/graphs: [Yes/No] 
        - Has images/figures: [Yes/No]
        
        **QUALITY ASSESSMENT:**
        - Content clarity: [Excellent/Good/Fair/Poor]
        - Readability: [Excellent/Good/Fair/Poor]
        - Information completeness: [Complete/Partial/Minimal]
        
        **FOLDER-SPECIFIC INSIGHTS:** [Provide insights specific to the document type/folder - what makes this page particularly relevant for this category]
        
        **ENHANCED MARKDOWN CONTENT:**
        [Convert the content to markdown format with the following structure:
        
        # Page {page_number}: [Title or Main Topic]
        
        ## 📋 Key Information Summary
        
        ### 🎯 Important Facts
        - [List important facts as bullet points]
        
        ### 📊 Key Data Points  
        - [List numerical data, dates, amounts as bullet points]
        
        ### Hand Written text
        - [List any hand-written text or annotations]

        ### 👥 People & Organizations
        - [List people and organizations with their roles/context]
        
        ### 📅 Important Dates & Deadlines
        - [List dates with their context/significance]
        
        ### ⚡ Action Items & Requirements
        - [List action items, tasks, requirements]
        
        ## 📄 Full OCR Content
        
        [Include the complete OCR-extracted text content here, properly formatted in markdown]
        
        ---
        
        > **Page Analysis:** [Page type] | **Quality:** [Confidence level] | **Processing time:** [If available]
        ]
        
        Be thorough and extract all meaningful information while maintaining clarity and organization.
        """

# Keep all the existing helper functions and classes unchanged
class APIError(Exception):
    """Custom exception for API errors"""
    def __init__(self, message: str, status_code: int = None, retry_after: int = None):
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(self.message)

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Retry decorator factory with exponential backoff"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not is_retryable_error(e):
                        logger.error(f"Non-retryable error: {e}")
                        raise e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                        break
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is worth retrying"""
    error_str = str(error).lower()
    retryable_indicators = [
        '504', '503', '502', '429', '500',
        'service unavailable', 'timeout', 'connection', 'network',
        'temporarily unavailable'
    ]
    return any(indicator in error_str for indicator in retryable_indicators)

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance image quality for better OCR results"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[str]:
    """Convert PDF pages to images using PyMuPDF"""
    logger.info(f"Converting PDF to images: {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        image = enhance_image_for_ocr(image)
        
        image_path = os.path.join(output_dir, f"page_{page_num + 1:04d}.png")
        image.save(image_path, "PNG", optimize=True, quality=95)
        image_paths.append(image_path)
        logger.info(f"Saved page {page_num + 1} as {image_path}")
    
    pdf_document.close()
    return image_paths

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clean_repetitive_content(content: str) -> str:
    """Clean up repetitive content and OCR artifacts"""
    lines = content.split('\n')
    cleaned_lines = []
    previous_line = ""
    repetition_count = 0
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if repetition_count > 0:
                cleaned_lines.append(f"*(above pattern repeated {repetition_count} more times)*")
                repetition_count = 0
            cleaned_lines.append(line)
            previous_line = ""
            continue
        
        if line == previous_line:
            repetition_count += 1
            if repetition_count == 1:
                continue
            elif repetition_count == 2:
                cleaned_lines.append(f"*(pattern repeated multiple times)*")
            continue
        else:
            if repetition_count > 0:
                repetition_count = 0
            cleaned_lines.append(line)
            previous_line = line
    
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = re.sub(r'^\d+\.\s*0{10,}\d{1,10}$', '*(sequential number pattern - likely OCR artifact)*', cleaned_content, flags=re.MULTILINE)
    cleaned_content = re.sub(r'(\b\w+\b)(\s*\1){10,}', r'\1 *(repeated multiple times)*', cleaned_content)
    
    return cleaned_content

# --- Enhanced Mistral API Pixtral Processing Functions ---
class EnhancedMistralPixtralProcessor:
    def __init__(self, api_key: str, model_name: str = "pixtral-large-latest"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = Mistral(api_key=api_key)
        self.prompt_generator = FolderPromptGenerator()
    
    @retry_with_backoff(max_retries=5, base_delay=4.0)
    def analyze_page_image(self, image_path: str, page_number: int, file_name: str, folder_type: FolderType) -> ProcessedPage:
        """Analyze a single page image using folder-specific prompts"""
        start_time = time.time()
        
        # Generate appropriate prompt for the folder type
        analysis_prompt = self.prompt_generator.generate_prompt(folder_type, page_number, file_name)
        
        base64_image = encode_image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": analysis_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                max_tokens=8192,  # Increased for more detailed output
                temperature=0.1,
                timeout_ms= 1800000,  # 10 min timeout
            )
            
            content = response.choices[0].message.content
            
            metadata = self._parse_enhanced_analysis_response(content, page_number, file_name, folder_type)
            markdown_content = self._extract_enhanced_markdown_content(content, metadata)
            
            processing_time = time.time() - start_time
            
            return ProcessedPage(
                metadata=metadata,
                markdown_content=markdown_content,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Error analyzing page {page_number}: {str(e)}"
            logger.error(error_msg)
            raise APIError(error_msg)

    def _parse_enhanced_analysis_response(self, content: str, page_number: int, file_name: str, folder_type: FolderType) -> PageMetadata:
        """Parse the enhanced analysis response to extract comprehensive metadata"""
        # Initialize with defaults
        metadata_dict = {
            'page_number': page_number,
            'file_name': file_name,
            'folder_type': folder_type,
            'page_type': PageType.TEXT,
            'title': None,
            'summary': "",
            'key_topics': [],
            'important_facts': [],
            'numerical_data': [],
            'action_items': [],
            'people_mentioned': [],
            'organizations': [],
            'dates_mentioned': [],
            'has_tables': False,
            'has_charts': False,
            'has_images': False,
            'confidence_score': 0.0,
            'readability_score': 0.0,
            'folder_specific_data': {}
        }

        try:
            content = clean_repetitive_content(content)
            
            # Parse each section
            sections = {
                'title': '**TITLE:**',
                'page_type': '**PAGE TYPE:**',
                'summary': '**SUMMARY:**',
                'key_topics': '**KEY TOPICS:**',
                'important_facts': '**IMPORTANT FACTS:**',
                'numerical_data': '**NUMERICAL DATA:**',
                'action_items': '**ACTION ITEMS:**',
                'people_mentioned': '**PEOPLE MENTIONED:**',
                'organizations': '**ORGANIZATIONS:**',
                'dates_mentioned': '**DATES MENTIONED:**',
                'visual_elements': '**VISUAL ELEMENTS:**',
                'quality_assessment': '**QUALITY ASSESSMENT:**',
                'folder_specific': '**FOLDER-SPECIFIC INSIGHTS:**'
            }
            
            for key, marker in sections.items():
                if marker in content:
                    section_content = content.split(marker)[1].split('**')[0].strip()
                    
                    if key == 'title':
                        if section_content and section_content != "No clear title":
                            metadata_dict['title'] = section_content
                    
                    elif key == 'page_type':
                        type_str = section_content.lower()
                        for ptype in PageType:
                            if ptype.value in type_str:
                                metadata_dict['page_type'] = ptype
                                break
                    
                    elif key == 'summary':
                        metadata_dict['summary'] = section_content
                    
                    elif key == 'key_topics':
                        if section_content:
                            metadata_dict['key_topics'] = [topic.strip() for topic in section_content.split(',') if topic.strip()]
                    
                    elif key in ['important_facts', 'action_items', 'people_mentioned', 'organizations']:
                        if section_content and '||' in section_content:
                            metadata_dict[key] = [item.strip() for item in section_content.split('||') if item.strip()]
                        elif section_content:
                            metadata_dict[key] = [section_content]
                    
                    elif key == 'numerical_data':
                        if section_content and '||' in section_content:
                            numerical_items = []
                            for item in section_content.split('||'):
                                if ':' in item:
                                    label, value = item.split(':', 1)
                                    numerical_items.append({'label': label.strip(), 'value': value.strip()})
                            metadata_dict['numerical_data'] = numerical_items
                    
                    elif key == 'dates_mentioned':
                        if section_content and '||' in section_content:
                            metadata_dict['dates_mentioned'] = [date.strip() for date in section_content.split('||') if date.strip()]
                    
                    elif key == 'visual_elements':
                        metadata_dict['has_tables'] = 'tables: yes' in section_content.lower()
                        metadata_dict['has_charts'] = 'charts/graphs: yes' in section_content.lower()
                        metadata_dict['has_images'] = 'images/figures: yes' in section_content.lower()
                    
                    elif key == 'quality_assessment':
                        # Parse quality metrics
                        quality_lower = section_content.lower()
                        
                        # Content clarity to confidence score
                        if 'excellent' in quality_lower:
                            metadata_dict['confidence_score'] = 0.95
                        elif 'good' in quality_lower:
                            metadata_dict['confidence_score'] = 0.8
                        elif 'fair' in quality_lower:
                            metadata_dict['confidence_score'] = 0.6
                        else:
                            metadata_dict['confidence_score'] = 0.4
                        
                        # Readability score
                        if 'readability: excellent' in quality_lower:
                            metadata_dict['readability_score'] = 0.95
                        elif 'readability: good' in quality_lower:
                            metadata_dict['readability_score'] = 0.8
                        elif 'readability: fair' in quality_lower:
                            metadata_dict['readability_score'] = 0.6
                        else:
                            metadata_dict['readability_score'] = 0.4
                    
                    elif key == 'folder_specific':
                        metadata_dict['folder_specific_data'] = {'insights': section_content}

        except Exception as e:
            logger.warning(f"Error parsing enhanced analysis response: {e}")
            # Fallback parsing
            content_lower = content.lower()
            metadata_dict.update({
                'has_tables': 'table' in content_lower,
                'has_charts': any(word in content_lower for word in ['chart', 'graph', 'plot']),
                'has_images': 'image' in content_lower or 'figure' in content_lower,
                'summary': content[:200] + "..." if len(content) > 200 else content,
                'confidence_score': 0.5,
                'readability_score': 0.5
            })

        return PageMetadata(**metadata_dict)

    def _extract_enhanced_markdown_content(self, content: str, metadata: PageMetadata) -> str:
        """Extract and enhance markdown content with better formatting"""
        try:
            # First try to extract the structured markdown from the response
            if "**ENHANCED MARKDOWN CONTENT:**" in content:
                markdown_section = content.split("**ENHANCED MARKDOWN CONTENT:**")[1].strip()
                return clean_repetitive_content(markdown_section)
            
            # If no structured markdown found, create it from the metadata and content
            else:
                return self._create_enhanced_markdown_from_metadata(content, metadata)
                
        except Exception:
            # Fallback - create basic markdown from available data
            return self._create_enhanced_markdown_from_metadata(content, metadata)
    
    def _create_enhanced_markdown_from_metadata(self, raw_content: str, metadata: PageMetadata) -> str:
        """Create enhanced markdown from metadata when structured output isn't available"""
        
        title = metadata.title or f"Page {metadata.page_number}"
        
        markdown = f"""# Page {metadata.page_number}: {title}

## 📋 Key Information Summary

### 🎯 Important Facts
"""
        
        if metadata.important_facts:
            for fact in metadata.important_facts:
                markdown += f"- {fact}\n"
        else:
            markdown += "- No specific important facts identified\n"
        
        markdown += "\n### 📊 Key Data Points\n"
        if metadata.numerical_data:
            for data_point in metadata.numerical_data:
                label = data_point.get('label', 'Data')
                value = data_point.get('value', 'N/A')
                markdown += f"- **{label}:** {value}\n"
        else:
            markdown += "- No specific numerical data identified\n"
        
        markdown += "\n### 👥 People & Organizations\n"
        if metadata.people_mentioned or metadata.organizations:
            if metadata.people_mentioned:
                markdown += "**People:**\n"
                for person in metadata.people_mentioned:
                    markdown += f"- {person}\n"
            if metadata.organizations:
                markdown += "**Organizations:**\n"
                for org in metadata.organizations:
                    markdown += f"- {org}\n"
        else:
            markdown += "- No specific people or organizations identified\n"
        
        markdown += "\n### 📅 Important Dates & Deadlines\n"
        if metadata.dates_mentioned:
            for date in metadata.dates_mentioned:
                markdown += f"- {date}\n"
        else:
            markdown += "- No specific dates identified\n"
        
        markdown += "\n### ⚡ Action Items & Requirements\n"
        if metadata.action_items:
            for item in metadata.action_items:
                markdown += f"- {item}\n"
        else:
            markdown += "- No specific action items identified\n"
        
        markdown += "\n## 📄 Full OCR Content\n\n"
        
        # Extract raw OCR content (try to clean it from the analysis)
        ocr_content = raw_content
        if "**ENHANCED MARKDOWN CONTENT:**" in raw_content:
            # If there was a markdown section, extract everything before it as OCR content
            ocr_content = raw_content.split("**ENHANCED MARKDOWN CONTENT:**")[0]
        
        # Clean and format the OCR content
        ocr_content = clean_repetitive_content(ocr_content)
        markdown += ocr_content
        
        markdown += f"""

---

> **Page Analysis:** {metadata.page_type.value.title()} | **Quality:** {metadata.confidence_score:.1%} | **Folder:** {metadata.folder_type.value.title()}
"""
        
        return markdown

# Keep all existing utility functions unchanged
def save_progress(progress_file: str, completed_pages: List[int], total_pages: int):
    """Save processing progress to a file"""
    progress_data = {
        'completed_pages': completed_pages,
        'total_pages': total_pages,
        'timestamp': time.time()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress(progress_file: str) -> Tuple[List[int], int]:
    """Load processing progress from a file"""
    if not os.path.exists(progress_file):
        return [], 0
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        completed_pages = progress_data.get('completed_pages', [])
        total_pages = progress_data.get('total_pages', 0)
        return completed_pages, total_pages
    except Exception as e:
        logger.warning(f"Could not load progress file: {e}")
        return [], 0

def process_pdf_with_enhanced_mistral_pixtral(
    pdf_path: str,
    api_key: str,
    model_name: str = "pixtral-large-latest",
    output_dir: str = "pdf_analysis_output",
    dpi: int = 300,
    folder: str = "others",
    folder_id: Optional[int] = None,
    resume_from_progress: bool = True,
    progress_file: str = "pdf_mistral_progress.json"
) -> List[str]:
    """
    Enhanced PDF processing with folder-specific prompts and comprehensive metadata
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "page_images")
    
    processor = EnhancedMistralPixtralProcessor(api_key, model_name)
    file_name = os.path.basename(pdf_path)
    
    # Determine folder type from ID or name
    if folder_id is not None:
        folder_type = FolderPromptGenerator.get_folder_type_by_id(folder_id)
        logger.info(f"Processing PDF with folder ID {folder_id} -> {folder_type.value}")
    else:
        folder_type = FolderPromptGenerator.get_folder_type(folder)
        logger.info(f"Processing PDF with folder type: {folder_type.value}")
    
    logger.info("Converting PDF to images...")
    image_paths = pdf_to_images(pdf_path, images_dir, dpi)
    total_pages = len(image_paths)
    
    # Progress handling
    completed_pages = []
    if resume_from_progress:
        completed_pages, saved_total_pages = load_progress(progress_file)
        if saved_total_pages != total_pages:
            logger.warning("PDF page count changed. Starting fresh.")
            completed_pages = []

    processed_pages = []
    failed_pages = []
    all_markdown_pages = [None] * total_pages

    try:
        for page_num, image_path in enumerate(image_paths, 1):
            if page_num in completed_pages:
                logger.info(f"Skipping already completed page {page_num}")
                continue

            logger.info(f"\n--- Processing page {page_num}/{total_pages} ---")
            
            try:
                processed_page = processor.analyze_page_image(
                    image_path=image_path,
                    page_number=page_num,
                    file_name=file_name,
                    folder_type=folder_type
                )
                
                processed_pages.append(processed_page)
                all_markdown_pages[page_num - 1] = processed_page.markdown_content
                
                completed_pages.append(page_num)
                save_progress(progress_file, completed_pages, total_pages)
                
                # Enhanced logging
                logger.info(f"Page {page_num} processed successfully in {processed_page.processing_time:.2f}s")
                logger.info(f"Folder type: {processed_page.metadata.folder_type.value}")
                logger.info(f"Page type: {processed_page.metadata.page_type.value}")
                logger.info(f"Confidence: {processed_page.metadata.confidence_score:.1%}")
                logger.info(f"Important facts: {len(processed_page.metadata.important_facts)}")
                logger.info(f"Data points: {len(processed_page.metadata.numerical_data)}")
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                failed_pages.append((page_num, str(e)))
                continue

        # Save enhanced results
        results_file = os.path.join(output_dir, "enhanced_processed_pages.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([page.dict() for page in processed_pages], f, indent=2, ensure_ascii=False)

        # Save all markdown pages as individual files
        markdown_dir = os.path.join(output_dir, "markdown_pages")
        os.makedirs(markdown_dir, exist_ok=True)
        
        for i, markdown_content in enumerate(all_markdown_pages):
            if markdown_content:
                page_file = os.path.join(markdown_dir, f"page_{i+1:04d}.md")
                with open(page_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

        # Create enhanced summary report
        create_enhanced_summary_report(processed_pages, output_dir, file_name, folder_type)

        logger.info(f"\nEnhanced processing complete!")
        logger.info(f"Successfully processed: {len(processed_pages)} pages")
        logger.info(f"Failed pages: {len(failed_pages)}")
        logger.info(f"Individual markdown files saved to: {markdown_dir}")
        
        # Clean up
        if len(completed_pages) == total_pages and not failed_pages:
            if os.path.exists(progress_file):
                os.remove(progress_file)
                logger.info("Processing completed successfully. Progress file removed.")

        final_markdown_pages = [page for page in all_markdown_pages if page is not None]
        return final_markdown_pages

    finally:
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            logger.info(f"Temporary images directory {images_dir} removed.")

def create_enhanced_summary_report(processed_pages: List[ProcessedPage], output_dir: str, file_name: str, folder_type: FolderType):
    """Create an enhanced summary report with folder-specific insights"""
    
    # Enhanced markdown report
    markdown_file = os.path.join(output_dir, "enhanced_analysis.md")
    with open(markdown_file, 'w', encoding='utf-8') as f:        
        f.write(f"# 📊 Enhanced Analysis of {file_name}\n\n")
        f.write(f"**📁 Document Type:** {folder_type.value.title()}\n")
        f.write(f"**📄 Total Pages:** {len(processed_pages)}\n")
        f.write(f"**🕐 Processing Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive summary with enhanced metrics
        f.write("## 🎯 Executive Summary\n\n")
        high_conf_pages = [p for p in processed_pages if p.metadata.confidence_score >= 0.8]
        avg_confidence = sum(p.metadata.confidence_score for p in processed_pages) / len(processed_pages)
        avg_readability = sum(p.metadata.readability_score for p in processed_pages) / len(processed_pages)
        
        total_facts = sum(len(p.metadata.important_facts) for p in processed_pages)
        total_data_points = sum(len(p.metadata.numerical_data) for p in processed_pages)
        total_action_items = sum(len(p.metadata.action_items) for p in processed_pages)
        
        f.write(f"This **{folder_type.value}** document contains **{len(processed_pages)}** pages with an average confidence score of **{avg_confidence:.1%}** ")
        f.write(f"and readability score of **{avg_readability:.1%}**. ")
        f.write(f"**{len(high_conf_pages)}** pages have high confidence scores (≥80%).\n\n")
        
        f.write(f"**📈 Data Extraction Summary:**\n")
        f.write(f"- **{total_facts}** important facts identified\n")
        f.write(f"- **{total_data_points}** numerical data points extracted\n")
        f.write(f"- **{total_action_items}** action items found\n\n")
        
        # Create combined markdown file with all pages
        combined_markdown_file = os.path.join(output_dir, "complete_document.md")
        with open(combined_markdown_file, 'w', encoding='utf-8') as combined_f:
            combined_f.write(f"# Complete Document: {file_name}\n\n")
            combined_f.write(f"**Document Type:** {folder_type.value.title()}\n")
            combined_f.write(f"**Total Pages:** {len(processed_pages)}\n")
            combined_f.write(f"**Processing Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            combined_f.write("---\n\n")
            
            for page in processed_pages:
                combined_f.write(page.markdown_content)
                combined_f.write("\n\n---\n\n")

        logger.info(f"Enhanced summary report saved to: {markdown_file}")
        logger.info(f"Complete document markdown saved to: {combined_markdown_file}")

# Example Usage
if __name__ == "__main__":
    # Configuration
    pdf_path = "privacy-policy-template.pdf"  # Your PDF file
    api_key = "C6QpntimbPnq11aBQuUc6zL0M7VCNHdG"  # Your Mistral API key
    
    # Define the folder type - this will determine the analysis approach
    folder = "contracts"  # Options: contracts, offers, invoices, sales_reports, others
    
    try:
        print("="*60)
        print(f"ENHANCED PDF PROCESSING WITH IMPORTANT DATA EXTRACTION")
        print("="*60)

        processed_pages = process_pdf_with_enhanced_mistral_pixtral(
            pdf_path=pdf_path,
            api_key=api_key,
            model_name="pixtral-large-latest",
            output_dir=f"enhanced_pdf_analysis_{folder}",
            dpi=300,
            folder=folder,
            resume_from_progress=True
        )
        
        print(f"\n✅ Enhanced processing complete! Analyzed {len(processed_pages)} pages.")
        print(f"📁 Results saved to: enhanced_pdf_analysis_{folder}/")
        print(f"📄 Individual page markdown files: enhanced_pdf_analysis_{folder}/markdown_pages/")
        print(f"📋 Complete document: enhanced_pdf_analysis_{folder}/complete_document.md")
        print(f"📊 Analysis report: enhanced_pdf_analysis_{folder}/enhanced_analysis.md")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        print(f"❌ Error processing PDF: {e}")