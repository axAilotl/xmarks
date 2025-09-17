"""
Transcript LLM Processor - Processes YouTube transcripts with LLM to format them properly
Combines fragmented sentences into coherent paragraphs and removes timestamps
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional, List, Dict

from core.config import config
from core.llm_interface import LLMInterface
from core.llm_cache import llm_cache
from core.metadata_db import get_metadata_db
from core.pipeline_registry import PipelineStage, pipeline_registry, register_pipeline_stages

logger = logging.getLogger(__name__)


def _transcript_stage_active(cfg) -> bool:
    """Stage predicate ensuring transcript formatting is permitted."""
    return bool(cfg.get('youtube.enable_llm_transcript_processing', True))


PIPELINE_STAGES = (
    PipelineStage(
        name='llm_processing.transcript_formatting',
        config_path='llm_processing.transcript_formatting',
        description='Format transcripts with LLM post-processing.',
        processor='TranscriptLLMProcessor',
        capabilities=('llm', 'transcript'),
        required_config=('processing.enable_llm_features', 'llm.tasks.transcript.enabled'),
        config_keys=(
            'processing.enable_llm_features',
            'llm.tasks.transcript.enabled',
            'youtube.enable_llm_transcript_processing',
            'youtube.transcript_chunk_size'
        ),
        predicate=_transcript_stage_active
    ),
)


register_pipeline_stages(*PIPELINE_STAGES)


class TranscriptLLMProcessor:
    """Processes YouTube transcripts using LLM to format them into readable paragraphs"""
    
    def __init__(self):
        self.config = config
        self.llm_interface = None
        # Stage controls consolidate llm task + youtube specific toggles
        self.enabled = pipeline_registry.is_enabled('llm_processing.transcript_formatting')
        self.chunk_size = config.get('youtube.transcript_chunk_size', 75000)
        self.metadata_db = None

        if self.enabled:
            try:
                # Get the full LLM config for initialization
                llm_config = config.get('llm', {})
                self.llm_interface = LLMInterface(llm_config)
                logger.info(f"Transcript LLM processor initialized with chunk size: {self.chunk_size}")
            except Exception as e:
                logger.error(f"Failed to initialize transcript LLM processor: {e}")
                self.enabled = False

        if self.enabled and config.get('database.enabled', False):
            try:
                self.metadata_db = get_metadata_db()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Transcript chunk cache unavailable: {exc}")
                self.metadata_db = None
    
    async def process_transcript(self, raw_transcript: str, context_id: Optional[str] = None) -> Optional[dict]:
        """
        Process a raw transcript with timestamps into formatted paragraphs with summary and tags
        For long transcripts, splits into chunks and processes each separately
        
        Args:
            raw_transcript: Raw transcript text with timestamps
            
        Returns:
            Dictionary with 'text', 'summary', and 'tags' keys, or None if processing fails
        """
        if not self.enabled or not self.llm_interface or not raw_transcript:
            return None
        
        try:
            route = self.llm_interface.resolve_task_route('transcript') if self.llm_interface else None
            if not route:
                logger.warning("Transcript LLM task disabled or no provider available")
                return None

            provider, model, _ = route
            prompt = config.get('youtube.transcript_processing_prompt', 
                              "Process the following transcript. Combine fragmented sentences into coherent paragraphs, remove all timestamps, and insert newlines between paragraphs where the context shifts. Do not edit the content beyond paragraph formation.\n\nReturn the result strictly as a JSON object with the following fields:\n- \"text\": the processed transcript in plain text paragraphs\n- \"summary\": a concise 2–4 sentence summary of the transcript\n- \"tags\": 3–8 relevant tags, returned as a single comma-separated string with no # symbols and no explanations\n\nReturn ONLY the JSON object, with no preamble or extra text.")
            cache_provider = f"{provider}:{model}" if provider or model else ""

            # Check if we need to chunk the transcript
            if len(raw_transcript) <= self.chunk_size:
                # Single chunk processing
                logger.info(f"Processing transcript with {provider} ({model}) - single chunk")
                return await self._process_single_chunk(
                    raw_transcript,
                    prompt,
                    provider,
                    model,
                    cache_provider,
                    context_id=context_id,
                    chunk_index=1
                )
            else:
                # Multi-chunk processing
                logger.info(f"Processing transcript with {provider} ({model}) - splitting into chunks of {self.chunk_size} chars")
                return await self._process_chunked_transcript(
                    raw_transcript,
                    prompt,
                    provider,
                    model,
                    cache_provider,
                    context_id=context_id
                )
                
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            return None
    
    async def _process_single_chunk(
        self,
        transcript_text: str,
        prompt: str,
        provider: str,
        model: str,
        cache_provider: str,
        context_id: Optional[str] = None,
        chunk_index: Optional[int] = None
    ) -> Optional[dict]:
        """Process a single chunk of transcript"""
        chunk_hash = self._hash_content(transcript_text)
        chunk_info = {
            'chunks_total': 1,
            'chunks_processed': 0,
            'chunks_failed': 0,
            'fallback_used': False,
            'failed_chunks': []
        }

        try:
            # Try cache first
            cached = llm_cache.get(transcript_text, 'transcript_fmt', cache_provider)
            if cached and all(k in cached for k in ['text', 'summary', 'tags']):
                logger.debug("Transcript LLM cache HIT for single chunk")
                if isinstance(cached, dict) and 'chunk_metadata' not in cached:
                    cached_meta = dict(chunk_info)
                    cached_meta['chunks_processed'] = 1
                    cached['chunk_metadata'] = cached_meta
                return cached

            response = await self.llm_interface.generate(
                prompt=transcript_text,
                system_prompt=prompt,
                provider=provider,
                model=model,
                max_tokens=16000,  # Large enough for long transcripts
                temperature=0.1   # Low temperature for consistent formatting
            )
            
            if response and not response.error and response.content:
                try:
                    raw = response.content.strip()
                    # Try direct JSON first
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        # Extract JSON object from noisy response (code fences, preambles)
                        cleaned = self._extract_json_object(raw)
                        result = json.loads(cleaned) if cleaned else None
                    
                    # Validate the JSON structure
                    if not result or not all(key in result for key in ['text', 'summary', 'tags']):
                        logger.error("Invalid JSON response: missing required fields")
                        # Minimal fallback: wrap raw response as text if parse failed
                        # This avoids dropping the chunk entirely
                        result = {
                            'text': raw if len(raw) < 100000 else raw[:99990],
                            'summary': '',
                            'tags': ''
                        }
                        chunk_info['fallback_used'] = True
                        chunk_info['chunks_failed'] = 1
                        if chunk_index is not None:
                            chunk_info['failed_chunks'] = [chunk_index]
                        self._record_chunk_failure(context_id, chunk_index, chunk_hash, cache_provider, 'invalid_json')

                    # Cache formatted chunk
                    try:
                        llm_cache.set(transcript_text, 'transcript_fmt', result, cache_provider)
                    except Exception:
                        pass

                    logger.info(f"✅ Successfully formatted single chunk: {len(transcript_text)} → {len(result['text'])} characters")
                    chunk_info['chunks_processed'] = 1 if not chunk_info['chunks_failed'] else 0
                    result['chunk_metadata'] = chunk_info
                    if context_id and chunk_index is not None and self.metadata_db:
                        try:
                            self.metadata_db.upsert_transcript_chunk(
                                context_id,
                                chunk_index,
                                chunk_hash,
                                json.dumps(result),
                                cache_provider
                            )
                        except Exception:
                            pass
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {response.content}")
                    chunk_info['fallback_used'] = True
                    chunk_info['chunks_failed'] = 1
                    if chunk_index is not None:
                        chunk_info['failed_chunks'] = [chunk_index]
                    self._record_chunk_failure(context_id, chunk_index, chunk_hash, cache_provider, 'json_decode_error')
                    return None
            else:
                error_msg = response.error if response else "No response received"
                logger.warning(f"❌ Single chunk formatting failed: {error_msg}")
                self._record_chunk_failure(context_id, chunk_index, chunk_hash, cache_provider, error_msg or 'no_response')
                return None
                
        except Exception as e:
            logger.error(f"Error processing single chunk: {e}")
            self._record_chunk_failure(context_id, chunk_index, chunk_hash, cache_provider, str(e))
            return None
    
    async def _process_chunked_transcript(
        self,
        raw_transcript: str,
        prompt: str,
        provider: str,
        model: str,
        cache_provider: str,
        context_id: Optional[str] = None
    ) -> Optional[dict]:
        """Process transcript in chunks and stitch results together"""
        try:
            # Split transcript into chunks at line boundaries
            chunks = self._split_transcript_into_chunks(raw_transcript)
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Process each chunk
            processed_chunks: List[dict] = []
            fallback_chunks: List[Tuple[int, str, str]] = []
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} characters)")

                chunk_hash = self._hash_content(chunk)
                cached_chunk = None
                if context_id and self.metadata_db:
                    cached_chunk = self._load_cached_chunk(context_id, i, chunk_hash, cache_provider)

                if cached_chunk:
                    processed_chunks.append(cached_chunk)
                    logger.info(f"♻️ Reused cached chunk {i}")
                    continue

                processed_chunk = await self._process_single_chunk(
                    chunk,
                    prompt,
                    provider,
                    model,
                    cache_provider,
                    context_id=context_id,
                    chunk_index=i
                )
                if processed_chunk:
                    processed_chunks.append(processed_chunk)
                    logger.info(f"✅ Chunk {i} processed successfully")
                else:
                    logger.warning(f"❌ Failed to process chunk {i}")
                    fallback_chunks.append((i, chunk, chunk_hash))
                    self._record_chunk_failure(context_id, i, chunk_hash, cache_provider, 'llm_failure')
                    # Continue with other chunks even if one fails

            if processed_chunks:
                # Combine text from all chunks
                combined_text_segments = [chunk['text'] for chunk in processed_chunks if chunk.get('text')]

                if fallback_chunks:
                    logger.warning(f"Using raw transcript text for {len(fallback_chunks)} chunk(s) due to LLM failures")
                    combined_text_segments.extend(chunk for _, chunk, _ in fallback_chunks)

                combined_text = '\n\n'.join(segment for segment in combined_text_segments if segment)

                # Create final result with combined text and aggregate summary/tags
                result = {
                    'text': combined_text,
                    'summary': self._combine_summaries([chunk['summary'] for chunk in processed_chunks]),
                    'tags': self._combine_tags([chunk['tags'] for chunk in processed_chunks])
                }

                chunk_metadata = {
                    'chunks_total': len(chunks),
                    'chunks_processed': len(processed_chunks),
                    'chunks_failed': len(fallback_chunks),
                    'fallback_used': bool(fallback_chunks),
                    'failed_chunks': [idx for idx, _, _ in fallback_chunks],
                }

                logger.info(f"✅ Successfully processed {len(processed_chunks)}/{len(chunks)} chunks: {len(raw_transcript)} → {len(result['text'])} characters")
                if context_id and self.metadata_db and not fallback_chunks:
                    try:
                        self.metadata_db.clear_transcript_chunks(context_id)
                    except Exception:
                        pass
                result['chunk_metadata'] = chunk_metadata
                return result
            elif fallback_chunks:
                logger.warning("Returning raw transcript fallback because all LLM chunk attempts failed")
                fallback_text = '\n\n'.join(chunk for _, chunk, _ in fallback_chunks)
                return {
                    'text': fallback_text,
                    'summary': '',
                    'tags': '',
                    'chunk_metadata': {
                        'chunks_total': len(chunks),
                        'chunks_processed': 0,
                        'chunks_failed': len(fallback_chunks),
                        'fallback_used': True,
                        'failed_chunks': [idx for idx, _, _ in fallback_chunks],
                    }
                }
            else:
                logger.error("❌ No chunks were processed successfully")
                return None
                
        except Exception as e:
            logger.error(f"Error processing chunked transcript: {e}")
            return None

    def _load_cached_chunk(self, context_id: str, chunk_index: int, expected_hash: str, cache_provider: str) -> Optional[Dict[str, str]]:
        """Load a cached transcript chunk from the metadata database when available."""
        if not self.metadata_db:
            return None
        try:
            cached = self.metadata_db.get_transcript_chunk(context_id, chunk_index)
        except Exception:
            return None

        if not cached:
            return None

        if cached.get('content_hash') != expected_hash:
            return None

        stored_provider = cached.get('model_provider') or ''
        if stored_provider and cache_provider and stored_provider != cache_provider:
            return None

        try:
            payload = cached.get('result_json') or ''
            data = json.loads(payload)
        except Exception:
            return None

        if not isinstance(data, dict) or not all(k in data for k in ('text', 'summary', 'tags')):
            return None

        return data

    def _hash_content(self, text: str) -> str:
        """Hash transcript content for cache validation."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _record_chunk_failure(
        self,
        context_id: Optional[str],
        chunk_index: Optional[int],
        content_hash: str,
        model_provider: str,
        reason: str
    ) -> None:
        """Persist chunk failure information when available."""
        if not self.metadata_db or not context_id or chunk_index is None:
            return
        try:
            payload = json.dumps({
                'status': 'failed',
                'reason': reason,
                'chunk_index': chunk_index
            })
            self.metadata_db.upsert_transcript_chunk(
                context_id,
                chunk_index,
                content_hash,
                payload,
                model_provider
            )
        except Exception:
            pass

    def _combine_summaries(self, summaries: List[str]) -> str:
        """Combine multiple summaries into a single coherent summary"""
        if not summaries:
            return ""
        
        # For now, just join with periods and clean up
        combined = ". ".join(summaries)
        # Remove any double periods
        combined = combined.replace("..", ".")
        # Ensure it ends with a period
        if not combined.endswith("."):
            combined += "."
        
        return combined
    
    def _combine_tags(self, tags_list: List[str]) -> str:
        """Combine multiple tag strings into a single comma-separated string"""
        if not tags_list:
            return ""
        
        # Split all tag strings and combine
        all_tags = []
        for tags_str in tags_list:
            if tags_str:
                # Split by comma and clean up
                tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
                all_tags.extend(tags)
        
        # Remove duplicates while preserving order
        unique_tags = []
        seen = set()
        for tag in all_tags:
            if tag.lower() not in seen:
                unique_tags.append(tag)
                seen.add(tag.lower())
        
        # Limit to 8 tags max
        return ", ".join(unique_tags[:8])
    
    def _split_transcript_into_chunks(self, transcript: str) -> List[str]:
        """Split transcript into chunks at line boundaries, respecting chunk size limit"""
        lines = transcript.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size, start a new chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _extract_json_object(self, text: str) -> Optional[str]:
        """Best-effort extraction of a JSON object from an LLM response.
        Strips code fences and grabs the outermost JSON braces.
        Applies minor fixes for trailing commas.
        """
        try:
            # Strip common code fences
            if text.startswith('```'):
                text = text.strip('`')
                # Remove potential language tag like ```json
                first_newline = text.find('\n')
                if first_newline != -1:
                    text = text[first_newline+1:]
            # Find outermost braces
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1 or end <= start:
                return None
            candidate = text[start:end+1]
            # Fix common JSON issues: trailing commas before } or ]
            import re
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            return candidate
        except Exception:
            return None
    
    def is_enabled(self) -> bool:
        """Check if transcript processing is enabled"""
        return self.enabled and self.llm_interface is not None
