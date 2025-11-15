use crate::docpack::{ContentEntry, Symbol};
use crate::vllm::VllmClient;
use crate::FileAst;
use anyhow::Result;
use serde_json;
use std::io::{self, Write};

/// System prompt for the LLM to generate documentation
const SYSTEM_PROMPT: &str = r#"
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

/// Response format we expect from the LLM
#[derive(Debug, serde::Deserialize)]
struct LlmResponse {
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

    /// Build the prompt for the LLM
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
            SYSTEM_PROMPT, file_ast.file_path, file_ast.language, truncated_source
        )
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
    batch_size: usize,
}

impl BatchProcessor {
    pub fn new(client: VllmClient, batch_size: usize) -> Self {
        Self {
            generator: DocGenerator::new(client),
            batch_size,
        }
    }

    /// Process a batch of files and return their documentation
    pub async fn process_files(
        &mut self,
        files: Vec<(FileAst, String)>,
    ) -> Result<Vec<ContentEntry>> {
        let total = files.len();
        let mut results = Vec::new();

        log_info(&format!(
            "Processing {} files in batches of {}",
            total, self.batch_size
        ));

        for (idx, (file_ast, source_code)) in files.into_iter().enumerate() {
            // Emit progress in a format the handler can parse
            log_progress(&format!(
                "Processing file {}/{}: {}",
                idx + 1,
                total,
                file_ast.file_path
            ));

            match self
                .generator
                .generate_file_docs(&file_ast, &source_code)
                .await
            {
                Ok(content) => {
                    let symbol_count = content.symbols.len();
                    log_info(&format!(
                        "✓ Generated {} symbols for {}",
                        symbol_count, file_ast.file_path
                    ));
                    results.push(content);
                }
                Err(e) => {
                    log_error(&format!(
                        "Failed to generate docs for {}: {}",
                        file_ast.file_path, e
                    ));
                    // Add empty entry on failure
                    results.push(ContentEntry {
                        path: file_ast.file_path.clone(),
                        symbols: Vec::new(),
                    });
                }
            }

            // Optional: add delay between requests to avoid rate limiting
            if idx < total - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        log_info(&format!(
            "Completed processing all {} files. Final token usage: {} input, {} output",
            total,
            self.generator.client.total_input_tokens,
            self.generator.client.total_output_tokens
        ));

        Ok(results)
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
