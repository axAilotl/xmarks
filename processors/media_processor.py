"""
Media Processor - Handles media downloads and linking
Downloads media from URLs and creates Obsidian-compatible links
"""

import logging
import requests
import hashlib
import os
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse
from core.data_models import Tweet, ProcessingStats
from core.config import config
from core.download_tracker import get_download_tracker
from core.pipeline_registry import PipelineStage, register_pipeline_stages

logger = logging.getLogger(__name__)


PIPELINE_STAGES = (
    PipelineStage(
        name='media_download',
        config_path='media_download',
        description='Download tweet media attachments and thumbnails.',
        processor='MediaProcessor',
        capabilities=('media',),
        config_keys=('paths.vault_dir', 'database.enabled')
    ),
)


register_pipeline_stages(*PIPELINE_STAGES)


class MediaProcessor:
    """Handles media downloads and Obsidian linking"""
    
    def __init__(self, images_dir: str = None, videos_dir: str = None):
        # Set up separate directories for images and videos
        vault_dir = Path(config.get('vault_dir', 'knowledge_vault'))
        
        if images_dir:
            self.images_dir = Path(images_dir)
        else:
            # Check if images_dir is an absolute path, otherwise make it relative to vault
            images_path = config.get('paths.images_dir', 'images')
            if Path(images_path).is_absolute():
                self.images_dir = Path(images_path)
            else:
                self.images_dir = vault_dir / images_path
        
        if videos_dir:
            self.videos_dir = Path(videos_dir)
        else:
            # Check if videos_dir is an absolute path, otherwise make it relative to vault
            videos_path = config.get('paths.videos_dir', 'videos')
            if Path(videos_path).is_absolute():
                self.videos_dir = Path(videos_path)
            else:
                self.videos_dir = vault_dir / videos_path
            
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep media_dir for backward compatibility (legacy single directory)
        self.media_dir = vault_dir / config.get('paths.media_dir', 'media')
        
        # Session for efficient HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Download tracker
        self.download_tracker = get_download_tracker()
        # Metadata DB
        self.metadata_db = None
        if config.get('database.enabled', False):
            try:
                from core.metadata_db import get_metadata_db
                self.metadata_db = get_metadata_db()
            except Exception:
                self.metadata_db = None
    
    def process_media(self, tweets: List[Tweet], resume: bool = True) -> ProcessingStats:
        """Process and download media for tweets"""
        stats = ProcessingStats()
        
        for tweet in tweets:
            try:
                if not tweet.media_items:
                    stats.skipped += 1
                    continue
                
                tweet_updated = False
                
                for i, media_item in enumerate(tweet.media_items, 1):
                    logger.debug(f"🧩 [MEDIA] Item {i} type={media_item.media_type} has_video={bool(getattr(media_item,'video_url', None))} tweet={tweet.id}")
                    if resume and media_item.downloaded:
                        continue
                    
                    # For videos, download both thumbnail and video
                    if media_item.media_type in ['video', 'animated_gif']:
                        # Download thumbnail first
                        if not self.download_tracker.is_404(media_item.media_url):
                            download_result = self._download_media(media_item, tweet.id, post_num=1, file_num=i, is_thumbnail=True, force_download=not resume)
                            if download_result:
                                tweet_updated = True
                                if hasattr(self, '_last_download_was_new') and self._last_download_was_new:
                                    stats.updated += 1
                                else:
                                    stats.skipped += 1
                        
                        # Download video if we have video URL
                        if media_item.video_url:
                            logger.debug(f"🎬 [MEDIA] Video URL for {tweet.id} #{i}: {media_item.video_url}")
                            # When resume is False, ignore cached 404s and retry
                            if (not resume) or (not self.download_tracker.is_404(media_item.video_url)):
                                video_result = self._download_video(media_item, tweet.id, post_num=1, file_num=i, force_download=not resume)
                            else:
                                logger.debug(f"⏭️ [MEDIA] Skip cached 404 (resume=True): {media_item.video_url}")
                            if video_result:
                                tweet_updated = True
                                if hasattr(self, '_last_video_download_was_new') and self._last_video_download_was_new:
                                    stats.updated += 1
                                else:
                                    stats.skipped += 1
                        else:
                            logger.debug(f"🚫 [MEDIA] No video_url on media item {media_item.media_id} (type={media_item.media_type}) for tweet {tweet.id}")
                    else:
                        # Regular photo/media download
                        if not self.download_tracker.is_404(media_item.media_url):
                            download_result = self._download_media(media_item, tweet.id, post_num=1, file_num=i, force_download=not resume)
                            if download_result:
                                tweet_updated = True
                                if hasattr(self, '_last_download_was_new') and self._last_download_was_new:
                                    stats.updated += 1
                                else:
                                    stats.skipped += 1
                            else:
                                stats.errors += 1
                
                # Attempt to recover missing video_filename from existing files
                try:
                    for j, media in enumerate(tweet.media_items, 1):
                        if media.media_type in ['video', 'animated_gif'] and not getattr(media, 'video_filename', None):
                            candidate = f"{tweet.id}_media_1_{j}.mp4"
                            if (self.media_dir / candidate).exists():
                                media.video_filename = candidate
                except Exception:
                    pass
                
                # Always try to replace t.co media URLs in content, regardless of download status
                self._replace_media_urls_in_content(tweet)
                
                stats.total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing media for tweet {tweet.id}: {e}")
                stats.errors += 1
        
        logger.info(f"📸 Media processing complete: {stats.updated} downloaded, {stats.skipped} skipped")
        return stats
    
    def _download_media(self, media_item, tweet_id: str, post_num: int = 1, file_num: int = 1, is_thumbnail: bool = False, force_download: bool = False) -> bool:
        """Download a single media item with tracking"""
        try:
            if not media_item.media_url:
                logger.warning(f"No media URL for media item {media_item.media_id}")
                return False
            
            url = media_item.media_url
            
            # Record pending download
            self.download_tracker.record_pending(url)
            
            # Generate filename with new standard naming
            if is_thumbnail and media_item.media_type in ['video', 'animated_gif']:
                # For video thumbnails, add _thumb suffix and save to images dir
                filename = self._generate_filename(url, 'photo', tweet_id, post_num, file_num, suffix='_thumb')
                filepath = self.images_dir / filename
            elif media_item.media_type in ['photo']:
                # Images go to images directory
                filename = self._generate_filename(url, media_item.media_type, tweet_id, post_num, file_num)
                filepath = self.images_dir / filename
            else:
                # Other media types (videos, gifs) go to videos directory
                filename = self._generate_filename(url, media_item.media_type, tweet_id, post_num, file_num)
                filepath = self.videos_dir / filename
            
            # Skip if already exists (unless force)
            if filepath.exists() and not force_download:
                file_size = filepath.stat().st_size
                media_item.filename = filename
                media_item.downloaded = True
                
                # Record success in tracker
                self.download_tracker.record_success(url, filename, str(filepath), file_size)
                logger.debug(f"Media already exists: {filename}")

                # Upsert file and download in DB
                if self.metadata_db:
                    try:
                        from core.metadata_db import FileMetadata, DownloadMetadata
                        from datetime import datetime
                        # vault-relative path if possible
                        vault_dir = Path(config.get('vault_dir', 'knowledge_vault'))
                        try:
                            rel_path = filepath.relative_to(vault_dir)
                        except Exception:
                            rel_path = filepath
                        self.metadata_db.upsert_file(FileMetadata(
                            path=str(rel_path),
                            file_type="media",
                            size_bytes=file_size,
                            updated_at=datetime.now().isoformat(),
                            source_id=tweet_id
                        ))
                        self.metadata_db.upsert_download(DownloadMetadata(
                            url=url,
                            status="success",
                            target_path=str(rel_path),
                            size_bytes=file_size
                        ))
                    except Exception:
                        pass
                
                # Flag that this was not a new download
                self._last_download_was_new = False
                return True
            
            # Download the media
            logger.debug(f"⬇️ [MEDIA] Download media: {url}")
            response = self.session.get(url, timeout=30)
            
            # Check for 404
            if response.status_code == 404:
                self.download_tracker.record_404(url, f"404 Not Found: {response.reason}")
                logger.warning(f"Media URL returned 404: {url}")
                return False
            
            response.raise_for_status()
            
            # Write to file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Update media item
            media_item.filename = filename
            media_item.downloaded = True
            
            # Record successful download
            file_size = len(response.content)
            self.download_tracker.record_success(url, filename, str(filepath), file_size)
            
            # Flag that this was a new download
            self._last_download_was_new = True
            
            logger.debug(f"✅ [MEDIA] Downloaded: {filename} ({file_size} bytes)")

            # Upsert file and download in DB
            if self.metadata_db:
                try:
                    from core.metadata_db import FileMetadata, DownloadMetadata
                    from datetime import datetime
                    vault_dir = Path(config.get('vault_dir', 'knowledge_vault'))
                    try:
                        rel_path = filepath.relative_to(vault_dir)
                    except Exception:
                        rel_path = filepath
                    self.metadata_db.upsert_file(FileMetadata(
                        path=str(rel_path),
                        file_type="media",
                        size_bytes=file_size,
                        updated_at=datetime.now().isoformat(),
                        source_id=tweet_id
                    ))
                    self.metadata_db.upsert_download(DownloadMetadata(
                        url=url,
                        status="success",
                        target_path=str(rel_path),
                        size_bytes=file_size
                    ))
                except Exception:
                    pass
            return True
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 404:
                self.download_tracker.record_404(url, str(e))
                logger.warning(f"Media URL returned 404: {url}")
            else:
                self.download_tracker.record_error(url, str(e))
                logger.error(f"Failed to download media {url}: {e}")
            return False
        except Exception as e:
            self.download_tracker.record_error(url, str(e))
            logger.error(f"Failed to download media {url}: {e}")
            return False
    
    def _download_video(self, media_item, tweet_id: str, post_num: int = 1, file_num: int = 1, force_download: bool = False) -> bool:
        """Download video file separately from thumbnail"""
        try:
            if not media_item.video_url:
                logger.warning(f"No video URL for media item {media_item.media_id}")
                return False
            
            url = media_item.video_url
            
            # Record pending download
            self.download_tracker.record_pending(url)
            
            # Generate video filename - videos go to videos directory
            filename = self._generate_filename(url, media_item.media_type, tweet_id, post_num, file_num)
            filepath = self.videos_dir / filename
            
            # Skip if already exists (unless force)
            if filepath.exists() and not force_download:
                file_size = filepath.stat().st_size
                media_item.video_filename = filename
                
                # Record success in tracker
                self.download_tracker.record_success(url, filename, str(filepath), file_size)
                logger.debug(f"Video already exists: {filename}")
                
                # Flag that this was not a new download
                self._last_video_download_was_new = False
                return True
            
            # Download the video
            logger.debug(f"⬇️ [MEDIA] Downloading video: {url}")
            response = self.session.get(url, timeout=60)  # Longer timeout for videos
            
            # Check for 404
            if response.status_code == 404:
                self.download_tracker.record_404(url, f"404 Not Found: {response.reason}")
                logger.warning(f"Video URL returned 404: {url}")
                return False
            
            response.raise_for_status()
            
            # Write to file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Update media item
            media_item.video_filename = filename
            
            # Record successful download
            file_size = len(response.content)
            self.download_tracker.record_success(url, filename, str(filepath), file_size)
            
            # Flag that this was a new download
            self._last_video_download_was_new = True
            
            logger.debug(f"Downloaded video: {filename} ({file_size} bytes)")

            # Upsert file and download in DB
            if self.metadata_db:
                try:
                    from core.metadata_db import FileMetadata, DownloadMetadata
                    from datetime import datetime
                    vault_dir = Path(config.get('vault_dir', 'knowledge_vault'))
                    try:
                        rel_path = filepath.relative_to(vault_dir)
                    except Exception:
                        rel_path = filepath
                    self.metadata_db.upsert_file(FileMetadata(
                        path=str(rel_path),
                        file_type="media",
                        size_bytes=file_size,
                        updated_at=datetime.now().isoformat(),
                        source_id=tweet_id
                    ))
                    self.metadata_db.upsert_download(DownloadMetadata(
                        url=url,
                        status="success",
                        target_path=str(rel_path),
                        size_bytes=file_size
                    ))
                except Exception:
                    pass
            return True
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 404:
                self.download_tracker.record_404(url, str(e))
                logger.warning(f"Video URL returned 404: {url}")
            else:
                self.download_tracker.record_error(url, str(e))
                logger.error(f"Failed to download video {url}: {e}")
            return False
        except Exception as e:
            self.download_tracker.record_error(url, str(e))
            logger.error(f"Failed to download video {url}: {e}")
            return False
    
    def _generate_filename(self, media_url: str, media_type: str, tweet_id: str, post_num: int, file_num: int, suffix: str = '') -> str:
        """Generate a filename for the media using standard naming convention: tweetid_media_postnum_filenum.ext"""
        try:
            # Parse URL to get extension
            parsed = urlparse(media_url)
            path = parsed.path
            
            # Get extension from URL
            ext = os.path.splitext(path)[1]
            if not ext:
                # Default extensions by type
                ext_map = {
                    'photo': '.jpg',
                    'video': '.mp4', 
                    'animated_gif': '.gif'
                }
                ext = ext_map.get(media_type, '.jpg')
            
            # Generate filename using standard convention: tweetid_media_postnum_filenum.ext
            filename = f"{tweet_id}_media_{post_num}_{file_num}{suffix}{ext}"
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate filename for {media_url}: {e}")
            # Fallback with standard naming
            return f"{tweet_id}_media_{post_num}_{file_num}.jpg"
    
    def _replace_media_urls_in_content(self, tweet: Tweet):
        """Replace t.co media URLs in tweet content with Obsidian links"""
        if not tweet.full_text or not tweet.media_items:
            return
        
        updated_text = tweet.full_text
        replacements_made = 0
        
        for media_item in tweet.media_items:
            # If media has a filename (downloaded or not), try to replace the t.co URL
            if not media_item.filename:
                logger.debug(f"Skipping media item - no filename: {media_item.media_id}")
                continue
            
            # Each media item should have an original_url (the t.co link) 
            tco_url = getattr(media_item, 'original_url', None)
            logger.debug(f"Media item original_url: {tco_url}")
            
            if not tco_url:
                # If no original_url is set, try to find it in the URL mappings
                # Look for URL mappings that point to this media URL
                if tweet.url_mappings:
                    for url_mapping in tweet.url_mappings:
                        if url_mapping.expanded_url == media_item.media_url:
                            tco_url = url_mapping.short_url
                            logger.debug(f"Found t.co URL in mappings: {tco_url}")
                            break
            
            if tco_url and tco_url in updated_text:
                # Replace with Obsidian media link - but don't add extra ![[]] if media is already inlined
                # Just remove the t.co URL since media is added separately in content generation
                updated_text = updated_text.replace(tco_url, "").strip()
                replacements_made += 1
                logger.debug(f"Removed t.co media URL: {tco_url}")
        
        # Clean up any double spaces that might result from URL removal
        updated_text = ' '.join(updated_text.split())
        
        if replacements_made > 0:
            tweet.full_text = updated_text
            logger.debug(f"Media URL replacement complete: {replacements_made} URLs removed")
    
    def get_statistics(self) -> Dict:
        """Get media processing statistics"""
        image_files = list(self.images_dir.glob("*")) if self.images_dir.exists() else []
        video_files = list(self.videos_dir.glob("*")) if self.videos_dir.exists() else []
        
        total_files = len(image_files) + len(video_files)
        total_size = (sum(f.stat().st_size for f in image_files if f.is_file()) + 
                     sum(f.stat().st_size for f in video_files if f.is_file())) / (1024*1024)
        
        return {
            'total_media_files': total_files,
            'image_files': len(image_files),
            'video_files': len(video_files),
            'images_directory': str(self.images_dir),
            'videos_directory': str(self.videos_dir),
            'total_size_mb': total_size,
            'images_size_mb': sum(f.stat().st_size for f in image_files if f.is_file()) / (1024*1024),
            'videos_size_mb': sum(f.stat().st_size for f in video_files if f.is_file()) / (1024*1024)
        }
