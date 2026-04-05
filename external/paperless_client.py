import requests
import os
from urllib.parse import unquote
import logging

from utils.logger_setup import setup_logging, log_and_print, log_only, print_only, Colors

class PaperlessDownloader:
    def __init__(self, base_url, token, logger_instance: logging.Logger = None):
        """Initialize with base URL and API token"""
        self.logger = logger_instance if logger_instance else setup_logging()
        log_and_print(self.logger, 'INFO', "Initializing PaperlessDownloader", Colors.BLUE)
        
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Token {token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        log_only(self.logger, 'DEBUG', f"Base URL: {self.base_url}")
        log_only(self.logger, 'DEBUG', "API headers configured")
    
    def _extract_original_filename(self, content_disposition, document_id, remove_date_prefix=False):
        """Extract the original filename from content-disposition header"""
        log_only(self.logger, 'DEBUG', f"Extracting filename from content-disposition: {content_disposition}")
        
        if not content_disposition:
            filename = f"document_{document_id}.pdf"
            log_only(self.logger, 'DEBUG', f"No content-disposition found, using default: {filename}")
            return filename
        
        # Handle filename* (RFC 5987) for Unicode filenames
        if "filename*=" in content_disposition:
            filename_star = content_disposition.split("filename*=")[1].split(';')[0]
            filename = unquote(filename_star[7:]) if filename_star.upper().startswith("UTF-8''") else unquote(filename_star)
            log_only(self.logger, 'DEBUG', f"Extracted from filename*: {filename}")
        elif "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].split(';')[0].strip('"').strip("'")
            log_only(self.logger, 'DEBUG', f"Extracted from filename: {filename}")
        else:
            filename = f"document_{document_id}.pdf"
            log_only(self.logger, 'DEBUG', f"No filename found, using default: {filename}")
        
        # Clean up filename
        filename = os.path.basename(filename)
        
        # Remove date prefix if requested
        if remove_date_prefix:
            import re
            date_pattern = r'^\d{4}-\d{2}-\d{2}\s+'
            if re.match(date_pattern, filename):
                original_filename = filename
                filename = re.sub(date_pattern, '', filename)
                log_only(self.logger, 'DEBUG', f"Removed date prefix: {original_filename} → {filename}")
        
        log_only(self.logger, 'DEBUG', f"Final filename: {filename}")
        return filename
    
    def download_document(self, document_id, output_dir=".", custom_filename=None, remove_date_prefix=False):
        """Download a document by ID and return the filepath"""
        download_url = f"{self.base_url}/api/documents/{document_id}/download/"
        log_and_print(self.logger, 'INFO', f"Starting download for document {document_id}", Colors.BLUE)
        log_only(self.logger, 'DEBUG', f"Download URL: {download_url}")
        
        try:
            print_only(f"📥 Downloading document {document_id}...", Colors.CYAN)
            response = requests.get(download_url, headers=self.headers)
            
            log_only(self.logger, 'DEBUG', f"Response status: {response.status_code}")
            log_only(self.logger, 'DEBUG', f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                # Determine filename
                if custom_filename:
                    filename = custom_filename
                    log_only(self.logger, 'DEBUG', f"Using custom filename: {filename}")
                else:
                    content_disposition = response.headers.get('content-disposition', '')
                    filename = self._extract_original_filename(content_disposition, document_id, remove_date_prefix)
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                log_only(self.logger, 'DEBUG', f"Output directory ensured: {output_dir}")
                
                # Write file
                filepath = os.path.join(output_dir, filename)
                log_only(self.logger, 'DEBUG', f"Writing to filepath: {filepath}")
                log_only(self.logger, 'DEBUG', f"Content size: {len(response.content)} bytes")
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                log_and_print(self.logger, 'INFO', f"Document saved to: {filepath}", Colors.GREEN)
                print_only("✅ Download completed successfully", Colors.GREEN)
                return filepath
            
            elif response.status_code == 401:
                log_and_print(self.logger, 'ERROR', "Authentication failed - invalid token", Colors.RED)
                return None
            elif response.status_code == 404:
                log_and_print(self.logger, 'ERROR', f"Document {document_id} not found", Colors.RED)
                return None
            else:
                log_and_print(self.logger, 'ERROR', f"Download failed. Status: {response.status_code}", Colors.RED)
                log_only(self.logger, 'DEBUG', f"Response content: {response.text[:200]}...")
                return None
                
        except requests.exceptions.RequestException as e:
            log_and_print(self.logger, 'ERROR', f"Network error during download: {str(e)}", Colors.RED)
            return None
        except OSError as e:
            log_and_print(self.logger, 'ERROR', f"File system error: {str(e)}", Colors.RED)
            return None
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Unexpected error during download: {str(e)}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full error details: {repr(e)}")
            return None

    def delete_file(self, filepath):
        """Delete a file by filepath"""
        log_and_print(self.logger, 'INFO', f"Attempting to delete file: {os.path.basename(filepath)}", Colors.BLUE)
        log_only(self.logger, 'DEBUG', f"Full filepath: {filepath}")
        
        if os.path.exists(filepath):
            try:
                print_only(f"🗑️ Deleting file...", Colors.CYAN)
                os.remove(filepath)
                log_and_print(self.logger, 'INFO', f"File deleted successfully: {os.path.basename(filepath)}", Colors.GREEN)
                return True
            except OSError as e:
                log_and_print(self.logger, 'ERROR', f"Error deleting file {os.path.basename(filepath)}: {e}", Colors.RED)
                return False
        else:
            log_and_print(self.logger, 'WARNING', f"File not found for deletion: {os.path.basename(filepath)}", Colors.YELLOW)
            return False

    def get_document_info(self, document_id):
        """Get document information without downloading"""
        info_url = f"{self.base_url}/api/documents/{document_id}/"
        log_and_print(self.logger, 'INFO', f"Fetching info for document {document_id}", Colors.BLUE)
        log_only(self.logger, 'DEBUG', f"Info URL: {info_url}")
        
        try:
            print_only(f"ℹ️ Fetching document info...", Colors.CYAN)
            response = requests.get(info_url, headers=self.headers)
            log_only(self.logger, 'DEBUG', f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                doc_info = response.json()
                log_and_print(self.logger, 'INFO', f"Document info retrieved successfully", Colors.GREEN)
                log_only(self.logger, 'DEBUG', f"Document title: {doc_info.get('title', 'N/A')}")
                log_only(self.logger, 'DEBUG', f"Document created: {doc_info.get('created', 'N/A')}")
                return doc_info
            elif response.status_code == 401:
                log_and_print(self.logger, 'ERROR', "Authentication failed - invalid token", Colors.RED)
                return None
            elif response.status_code == 404:
                log_and_print(self.logger, 'ERROR', f"Document {document_id} not found", Colors.RED)
                return None
            else:
                log_and_print(self.logger, 'ERROR', f"Failed to get document info. Status: {response.status_code}", Colors.RED)
                return None
                
        except requests.exceptions.RequestException as e:
            log_and_print(self.logger, 'ERROR', f"Network error fetching document info: {str(e)}", Colors.RED)
            return None
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Unexpected error fetching document info: {str(e)}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full error details: {repr(e)}")
            return None

    def test_connection(self):
        """Test the connection to Paperless-ngx API"""
        test_url = f"{self.base_url}/api/documents/"
        log_and_print(self.logger, 'INFO', "Testing API connection", Colors.BLUE)
        log_only(self.logger, 'DEBUG', f"Test URL: {test_url}")
        
        try:
            print_only("🔄 Testing connection...", Colors.CYAN)
            response = requests.get(test_url, headers=self.headers, params={'page_size': 1})
            log_only(self.logger, 'DEBUG', f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                log_and_print(self.logger, 'INFO', "API connection successful", Colors.GREEN)
                print_only("✅ Connection test successful", Colors.GREEN)
                return True
            elif response.status_code == 401:
                log_and_print(self.logger, 'ERROR', "Authentication failed - invalid token", Colors.RED)
                return False
            else:
                log_and_print(self.logger, 'ERROR', f"Connection test failed. Status: {response.status_code}", Colors.RED)
                return False
                
        except requests.exceptions.RequestException as e:
            log_and_print(self.logger, 'ERROR', f"Network error during connection test: {str(e)}", Colors.RED)
            return False
        except Exception as e:
            log_and_print(self.logger, 'ERROR', f"Unexpected error during connection test: {str(e)}", Colors.RED)
            log_only(self.logger, 'DEBUG', f"Full error details: {repr(e)}")
            return False


def download_and_process_document(base_url: str, api_token: str, document_id: int, 
                                output_dir: str = ".", custom_filename: str = None, 
                                remove_date_prefix: bool = False, 
                                delete_after_processing: bool = False,
                                logger_instance: logging.Logger = None) -> str:
    """
    Convenience function to download and optionally delete a document
    
    Args:
        base_url: Paperless-ngx base URL
        api_token: API authentication token
        document_id: Document ID to download
        output_dir: Output directory for downloaded file
        custom_filename: Optional custom filename
        remove_date_prefix: Whether to remove date prefix from filename
        delete_after_processing: Whether to delete file after successful download
        logger_instance: Optional logger instance
    
    Returns:
        Downloaded filepath or None if failed
    """
    logger = logger_instance if logger_instance else setup_logging()
    log_and_print(logger, 'INFO', "Starting document download and processing", Colors.BLUE)
    
    try:
        # Create downloader instance
        downloader = PaperlessDownloader(base_url, api_token, logger)
        
        # Test connection first
        if not downloader.test_connection():
            log_and_print(logger, 'ERROR', "Connection test failed, aborting download", Colors.RED)
            return None
        
        # Download document
        filepath = downloader.download_document(
            document_id=document_id,
            output_dir=output_dir,
            custom_filename=custom_filename,
            remove_date_prefix=remove_date_prefix
        )
        
        if filepath:
            log_and_print(logger, 'INFO', f"Download successful: {os.path.basename(filepath)}", Colors.GREEN)
            
            # Optional: Delete after processing
            if delete_after_processing:
                log_and_print(logger, 'INFO', "Deleting file after processing", Colors.BLUE)
                downloader.delete_file(filepath)
                return None  # File was deleted
            
            return filepath
        else:
            log_and_print(logger, 'ERROR', "Download failed", Colors.RED)
            return None
            
    except Exception as e:
        log_and_print(logger, 'ERROR', f"Error in download and process: {str(e)}", Colors.RED)
        log_only(logger, 'DEBUG', f"Full error details: {repr(e)}")
        return None


# Usage example
if __name__ == "__main__":
    logger = setup_logging()
    
    # Configuration
    BASE_URL = "https://ocr.koelmann.eu"
    API_TOKEN = "db8a8aed53d882012bb21d393420564b765f2229"
    DOCUMENT_ID = 1001
    
    print_only("="*60, Colors.MAGENTA)
    print_only("📄 Paperless Document Downloader", Colors.BOLD + Colors.WHITE)
    print_only("="*60, Colors.MAGENTA)
    
    log_and_print(logger, 'INFO', "Starting Paperless downloader script", Colors.BLUE)
    
    try:
        # Create downloader instance
        downloader = PaperlessDownloader(BASE_URL, API_TOKEN, logger)
        
        # Test connection
        if downloader.test_connection():
            # Get document info first
            doc_info = downloader.get_document_info(DOCUMENT_ID)
            if doc_info:
                print_only(f"📋 Document: {doc_info.get('title', 'Unknown')}", Colors.BLUE)
            
            # Download document
            filepath = downloader.download_document(DOCUMENT_ID, remove_date_prefix=True)
            
            if filepath:
                log_and_print(logger, 'INFO', f"Successfully downloaded: {os.path.basename(filepath)}", Colors.GREEN)
                print_only(f"💾 File location: {filepath}", Colors.GREEN)
                
                # Process your file here
                print_only("🔄 Processing file...", Colors.CYAN)
                # ... your processing code ...
                
                # Optional: Delete the file after processing
                # downloader.delete_file(filepath)
            else:
                log_and_print(logger, 'ERROR', "Download failed", Colors.RED)
        else:
            log_and_print(logger, 'ERROR', "Connection test failed", Colors.RED)
            
        log_and_print(logger, 'INFO', "Script execution completed", Colors.GREEN)
        
    except Exception as e:
        log_and_print(logger, 'ERROR', f"Script execution failed: {str(e)}", Colors.RED)
        log_only(logger, 'DEBUG', f"Full error details: {repr(e)}")