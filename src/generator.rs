use crate::docpack::{ContentEntry, Symbol};
use crate::vllm::VllmClient;
use crate::FileAst;
use anyhow::Result;
use serde_json;
use std::io::{self, Write};

/// System prompt for the LLM to generate documentation for a single file
const SYSTEM_PROMPT_SINGLE: &str = r#"
You generate documentation strictly as JSON.

OUTPUT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanations.
- JSON must start with '{' and end with '}'.
- Use this exact structure:
  {
    "symbols": [
      {
        "id": "path::name",
        "kind": "function|method|class|struct|enum|trait|interface|constant|variable|module|type",
        "signature": "string",
        "summary": "one short sentence",
        "description": "one or two concise sentences",
        "examples": [],
        "complexity": "",
        "dependencies": [],
        "notes": ""
      }
    ]
  }
- Use double quotes everywhere.
- No trailing commas.
- If something does not exist, use empty array [] or empty string "".
- Never output multiple top-level objects. One single JSON object only.

Your task:
Read the provided source code and return documentation in this JSON format.
Generate only valid JSON and nothing else."#;

/// System prompt for the LLM to generate documentation for multiple files in a batch
const SYSTEM_PROMPT_BATCH: &str = r#"
You generate documentation strictly as JSON.

OUTPUT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanations.
- JSON must start with '{' and end with '}'.
- Use this exact structure for MULTIPLE files:
  {
    "files": [
      {
        "file_path": "exact file path from input",
        "symbols": [
          {
            "id": "path::name",
            "kind": "function|method|class|struct|enum|trait|interface|constant|variable|module|type",
            "signature": "string",
            "summary": "one short sentence",
            "description": "one or two concise sentences",
            "examples": [],
            "complexity": "",
            "dependencies": [],
            "notes": ""
          }
        ]
      }
    ]
  }
- Use double quotes everywhere.
- No trailing commas.
- If something does not exist, use empty array [] or empty string "".
- You must output documentation for ALL files provided.
- The "file_path" in your response must EXACTLY match the file path shown in the input.

Your task:
Read the provided source code files and return documentation for each file in this JSON format.
Generate only valid JSON and nothing else."#;

/// Response format we expect from the LLM for a single file
#[derive(Debug, serde::Deserialize)]
struct LlmResponse {
    symbols: Vec<Symbol>,
}

/// Response format for batch processing multiple files
#[derive(Debug, serde::Deserialize)]
struct LlmBatchResponse {
    files: Vec<LlmFileResponse>,
}

#[derive(Debug, serde::Deserialize)]
struct LlmFileResponse {
    file_path: String,
    symbols: Vec<Symbol>,
}

/// Generator that converts AST to documentation using vLLM
pub struct DocGenerator {
    client: VllmClient,
}

impl DocGenerator {
    pub fn new(client: VllmClient) -> Self {
        Self { client }
    }

    /// Generate documentation for a single file
    pub async fn generate_file_docs(
        &mut self,
        file_ast: &FileAst,
        source_code: &str,
    ) -> Result<ContentEntry> {
        log_info(&format!("Generating docs for: {}", file_ast.file_path));

        // Build prompt with file context
        let prompt = self.build_prompt(file_ast, source_code);

        // Call vLLM with optimized parameters for JSON output
        // Stop tokens help ensure valid JSON completion
        let response = self
            .client
            .generate_with_params(
                prompt,
                Some(0.3),  // Low temperature for more deterministic output
                Some(4096), // Increased max tokens for complex files
                Some(0.95), // Top-p sampling
                Some(vec!["\n\n---".to_string()]), // Stop token to prevent generating beyond JSON
            )
            .await?;

        // Log token usage for this file
        let (total_in, total_out) = self.client.get_token_stats();
        log_info(&format!(
            "[TOKENS] File: {} | Input: {} | Output: {} | Total: {}",
            file_ast.file_path,
            total_in,
            total_out,
            total_in + total_out
        ));

        let response_text = VllmClient::extract_text(&response);

        // Parse LLM response as JSON
        let symbols = match self.parse_llm_response(&response_text) {
            Ok(symbols) => symbols,
            Err(e) => {
                log_error(&format!(
                    "Failed to parse LLM response for {}: {}",
                    file_ast.file_path, e
                ));
                log_error(&format!("Raw response: {}", response_text));
                // Return empty symbols on parse failure
                Vec::new()
            }
        };

        Ok(ContentEntry {
            path: file_ast.file_path.clone(),
            symbols,
        })
    }

    /// Build JSON schema for guided decoding
    fn build_json_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string"},
                            "signature": {"type": "string"},
                            "summary": {"type": "string"},
                            "description": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "complexity": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "notes": {"type": "string"}
                        },
                        "required": ["id", "kind", "summary", "description"]
                    }
                }
            },
            "required": ["symbols"]
        })
    }

    /// Build the prompt for a single file
    fn build_prompt(&self, file_ast: &FileAst, source_code: &str) -> String {
        // Truncate source code if too long (keep first 8000 chars to stay within context)
        let truncated_source = if source_code.len() > 8000 {
            format!("{}...\n[truncated]", &source_code[..8000])
        } else {
            source_code.to_string()
        };

        // Build a compact representation focusing on the source code
        // Skip AST for now to reduce tokens - the model can infer from source
        format!(
            "{}\n\nFile: {}\nLanguage: {}\n\nCode:\n{}\n\nJSON:",
            SYSTEM_PROMPT_SINGLE, file_ast.file_path, file_ast.language, truncated_source
        )
    }

    /// Build a batch prompt for multiple files
    fn build_batch_prompt(&self, files: &[(FileAst, String)]) -> String {
        let mut prompt = format!("{}\n\n", SYSTEM_PROMPT_BATCH);

        for (idx, (file_ast, source_code)) in files.iter().enumerate() {
            // Truncate individual files if too long
            let truncated_source = if source_code.len() > 8000 {
                format!("{}...\n[truncated]", &source_code[..8000])
            } else {
                source_code.to_string()
            };

            prompt.push_str(&format!(
                "=== FILE {} ===\nPath: {}\nLanguage: {}\n\nCode:\n{}\n\n",
                idx + 1,
                file_ast.file_path,
                file_ast.language,
                truncated_source
            ));
        }

        prompt.push_str("JSON:");
        prompt
    }

    /// Estimate tokens for a file (rough heuristic: ~4 chars per token for code)
    fn estimate_tokens(&self, file_ast: &FileAst, source_code: &str) -> usize {
        // System prompt tokens
        let system_tokens = SYSTEM_PROMPT_SINGLE.len() / 4;

        // File metadata tokens
        let metadata_tokens = (file_ast.file_path.len() + file_ast.language.len() + 50) / 4;

        // Source code tokens (truncated to 8000 chars max)
        let source_len = source_code.len().min(8000);
        let source_tokens = source_len / 4;

        system_tokens + metadata_tokens + source_tokens
    }

    /// Generate documentation for multiple files in a single batch request
    pub async fn generate_batch_docs(
        &mut self,
        files: &[(FileAst, String)],
    ) -> Result<Vec<ContentEntry>> {
        if files.is_empty() {
            return Ok(Vec::new());
        }

        log_info(&format!("Generating docs for batch of {} files", files.len()));

        // Build batch prompt
        let prompt = self.build_batch_prompt(files);

        // Call vLLM with higher max_tokens for batch processing
        let max_tokens = (files.len() * 4096).min(32000) as u32;
        let response = self
            .client
            .generate_with_params(
                prompt,
                Some(0.3),  // Low temperature for deterministic output
                Some(max_tokens),
                Some(0.95), // Top-p sampling
                Some(vec!["\n\n---".to_string()]),
            )
            .await?;

        let response_text = VllmClient::extract_text(&response);

        // Parse batch response
        match self.parse_batch_response(&response_text, files) {
            Ok(results) => {
                log_info(&format!("✓ Successfully parsed batch of {} files", results.len()));
                Ok(results)
            }
            Err(e) => {
                log_error(&format!("Failed to parse batch response: {}", e));
                log_error(&format!("Raw response: {}", response_text));

                // Return empty entries for all files on failure
                Ok(files.iter().map(|(file_ast, _)| ContentEntry {
                    path: file_ast.file_path.clone(),
                    symbols: Vec::new(),
                }).collect())
            }
        }
    }

    /// Parse batch response with repair attempts
    fn parse_batch_response(
        &self,
        text: &str,
        files: &[(FileAst, String)],
    ) -> Result<Vec<ContentEntry>> {
        // Extract JSON content
        let json_text = self.extract_json_content(text);

        // Try parsing as batch response
        if let Ok(batch_response) = serde_json::from_str::<LlmBatchResponse>(json_text) {
            return self.map_batch_response_to_entries(batch_response, files);
        }

        // Try to repair and parse again
        let repaired = self.repair_json(json_text);
        if let Ok(batch_response) = serde_json::from_str::<LlmBatchResponse>(&repaired) {
            return self.map_batch_response_to_entries(batch_response, files);
        }

        // Try aggressive extraction
        if let Some(extracted) = self.extract_json_object(text) {
            if let Ok(batch_response) = serde_json::from_str::<LlmBatchResponse>(&extracted) {
                return self.map_batch_response_to_entries(batch_response, files);
            }
            let repaired_extracted = self.repair_json(&extracted);
            if let Ok(batch_response) = serde_json::from_str::<LlmBatchResponse>(&repaired_extracted) {
                return self.map_batch_response_to_entries(batch_response, files);
            }
        }

        // Final attempt
        let batch_response: LlmBatchResponse = serde_json::from_str(json_text)?;
        self.map_batch_response_to_entries(batch_response, files)
    }

    /// Map batch response to content entries, ensuring all files are represented
    fn map_batch_response_to_entries(
        &self,
        batch_response: LlmBatchResponse,
        files: &[(FileAst, String)],
    ) -> Result<Vec<ContentEntry>> {
        let mut entries = Vec::new();

        for (file_ast, _) in files {
            // Find matching response for this file
            let symbols = batch_response
                .files
                .iter()
                .find(|f| f.file_path == file_ast.file_path)
                .map(|f| f.symbols.clone())
                .unwrap_or_else(Vec::new);

            entries.push(ContentEntry {
                path: file_ast.file_path.clone(),
                symbols,
            });
        }

        Ok(entries)
    }

    /// Parse the LLM's JSON response with repair attempts
    fn parse_llm_response(&self, text: &str) -> Result<Vec<Symbol>> {
        // Step 1: Extract JSON from various formats
        let json_text = self.extract_json_content(text);

        // Step 2: Try parsing as-is
        if let Ok(response) = serde_json::from_str::<LlmResponse>(json_text) {
            return Ok(response.symbols);
        }

        // Step 3: Try to repair common JSON issues
        let repaired = self.repair_json(json_text);
        if let Ok(response) = serde_json::from_str::<LlmResponse>(&repaired) {
            return Ok(response.symbols);
        }

        // Step 4: Try aggressive extraction - find first { to last }
        if let Some(extracted) = self.extract_json_object(text) {
            if let Ok(response) = serde_json::from_str::<LlmResponse>(&extracted) {
                return Ok(response.symbols);
            }
            // Try repairing the extracted JSON
            let repaired_extracted = self.repair_json(&extracted);
            if let Ok(response) = serde_json::from_str::<LlmResponse>(&repaired_extracted) {
                return Ok(response.symbols);
            }
        }

        // Step 5: Final attempt - parse original and return error
        let response: LlmResponse = serde_json::from_str(json_text)?;
        Ok(response.symbols)
    }

    /// Extract JSON content from various wrapper formats
    fn extract_json_content<'a>(&self, text: &'a str) -> &'a str {
        let trimmed = text.trim();

        // Check for markdown code blocks
        if trimmed.contains("```json") {
            if let Some(json_part) = trimmed.split("```json").nth(1) {
                if let Some(content) = json_part.split("```").next() {
                    return content.trim();
                }
            }
        } else if trimmed.contains("```") {
            if let Some(code_part) = trimmed.split("```").nth(1) {
                if let Some(content) = code_part.split("```").next() {
                    return content.trim();
                }
            }
        }

        trimmed
    }

    /// Aggressively extract JSON object from text
    fn extract_json_object(&self, text: &str) -> Option<String> {
        let first_brace = text.find('{')?;
        let last_brace = text.rfind('}')?;

        if last_brace > first_brace {
            Some(text[first_brace..=last_brace].to_string())
        } else {
            None
        }
    }

    /// Attempt to repair common JSON formatting issues
    fn repair_json(&self, text: &str) -> String {
        let mut repaired = text.to_string();

        // Remove any trailing commas before closing brackets
        repaired = repaired.replace(",]", "]").replace(",}", "}");

        // If JSON is truncated mid-string, try to close it
        // Count open/close braces and brackets
        let open_braces = repaired.matches('{').count();
        let close_braces = repaired.matches('}').count();
        let open_brackets = repaired.matches('[').count();
        let close_brackets = repaired.matches(']').count();

        // If we're in an incomplete string (odd number of quotes), close it
        let quote_count = repaired.matches('"').count();
        if quote_count % 2 != 0 {
            repaired.push('"');
        }

        // Close any unclosed arrays
        for _ in 0..(open_brackets - close_brackets) {
            repaired.push(']');
        }

        // Close any unclosed objects
        for _ in 0..(open_braces - close_braces) {
            repaired.push('}');
        }

        repaired
    }

    /// Get token statistics
    pub fn get_token_stats(&self) -> (u64, u64) {
        self.client.get_token_stats()
    }
}

/// Batch processor for generating documentation for multiple files
pub struct BatchProcessor {
    generator: DocGenerator,
    max_tokens_per_batch: usize,
}

impl BatchProcessor {
    /// Create a new batch processor
    /// max_tokens_per_batch: Maximum tokens per batch (default: 130,000)
    pub fn new(client: VllmClient, max_tokens_per_batch: usize) -> Self {
        Self {
            generator: DocGenerator::new(client),
            max_tokens_per_batch,
        }
    }

    /// Group files into batches based on token limits
    fn group_files_into_batches(&self, files: Vec<(FileAst, String)>) -> Vec<Vec<(FileAst, String)>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_tokens = 0;

        // Reserve tokens for system prompt and output (roughly 2000 tokens)
        let system_overhead = 2000;

        for (file_ast, source_code) in files {
            let file_tokens = self.generator.estimate_tokens(&file_ast, &source_code);

            // Check if adding this file would exceed the limit
            if current_tokens + file_tokens + system_overhead > self.max_tokens_per_batch && !current_batch.is_empty() {
                // Start a new batch
                batches.push(current_batch);
                current_batch = Vec::new();
                current_tokens = 0;
            }

            current_batch.push((file_ast, source_code));
            current_tokens += file_tokens;
        }

        // Add the last batch if it's not empty
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    /// Process a batch of files and return their documentation
    pub async fn process_files(
        &mut self,
        files: Vec<(FileAst, String)>,
    ) -> Result<Vec<ContentEntry>> {
        let total_files = files.len();

        if total_files == 0 {
            return Ok(Vec::new());
        }

        log_info(&format!(
            "Processing {} files with max {} tokens per batch",
            total_files, self.max_tokens_per_batch
        ));

        // Group files into batches
        let batches = self.group_files_into_batches(files);
        let num_batches = batches.len();

        log_info(&format!(
            "Created {} batches (avg {:.1} files per batch)",
            num_batches,
            total_files as f64 / num_batches as f64
        ));

        let mut all_results = Vec::new();
        let mut files_processed = 0;

        for (batch_idx, batch) in batches.into_iter().enumerate() {
            let batch_size = batch.len();
            let batch_num = batch_idx + 1;

            log_info(&format!(
                "Processing batch {}/{} ({} files)...",
                batch_num, num_batches, batch_size
            ));

            // Emit progress for the first file in the batch
            if let Some((first_file, _)) = batch.first() {
                log_progress(&format!(
                    "Processing file {}/{}: {} (batch {}/{})",
                    files_processed + 1,
                    total_files,
                    first_file.file_path,
                    batch_num,
                    num_batches
                ));
            }

            // Process the batch
            match self.generator.generate_batch_docs(&batch).await {
                Ok(batch_results) => {
                    let total_symbols: usize = batch_results.iter().map(|e| e.symbols.len()).sum();
                    log_info(&format!(
                        "✓ Batch {}/{} complete: {} files, {} symbols",
                        batch_num, num_batches, batch_size, total_symbols
                    ));

                    files_processed += batch_size;
                    all_results.extend(batch_results);
                }
                Err(e) => {
                    log_error(&format!(
                        "Failed to process batch {}/{}: {}",
                        batch_num, num_batches, e
                    ));

                    // Add empty entries for all files in failed batch
                    for (file_ast, _) in batch {
                        all_results.push(ContentEntry {
                            path: file_ast.file_path.clone(),
                            symbols: Vec::new(),
                        });
                    }
                    files_processed += batch_size;
                }
            }

            // Small delay between batches to avoid rate limiting
            if batch_idx < num_batches - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        log_info(&format!(
            "Completed processing all {} files in {} batches. Final token usage: {} input, {} output",
            total_files,
            num_batches,
            self.generator.client.total_input_tokens,
            self.generator.client.total_output_tokens
        ));

        Ok(all_results)
    }

    pub fn get_token_stats(&self) -> (u64, u64) {
        self.generator.get_token_stats()
    }
}

/// Log an informational message to stderr (won't interfere with stdout)
fn log_info(message: &str) {
    eprintln!("[LOG] {}", message);
    let _ = io::stderr().flush();
}

/// Log progress message to stderr (for Python handler to parse)
fn log_progress(message: &str) {
    eprintln!("[PROGRESS] {}", message);
    let _ = io::stderr().flush();
}

/// Log an error message to stderr
fn log_error(message: &str) {
    eprintln!("[ERROR] {}", message);
    let _ = io::stderr().flush();
}
