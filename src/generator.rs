use crate::docpack::{ContentEntry, Symbol};
use crate::vllm::VllmClient;
use crate::FileAst;
use anyhow::Result;
use serde_json;
use std::io::{self, Write};

/// System prompt for the LLM to generate documentation
const SYSTEM_PROMPT: &str = r#"You are an expert code documentation assistant. Given source code and its AST, generate structured documentation for all symbols (functions, classes, methods, etc.).

Output ONLY valid JSON in this exact format:
{
  "symbols": [
    {
      "id": "file_path::symbol_name",
      "kind": "function|method|class|struct|enum|trait|interface|constant|variable|module|type",
      "signature": "full signature if applicable",
      "summary": "one-line summary",
      "description": "detailed explanation",
      "examples": ["code example 1", "code example 2"],
      "complexity": "O(n) or other complexity analysis",
      "dependencies": ["other::symbol::ids"],
      "notes": "additional context or caveats"
    }
  ]
}

Rules:
- Generate stable IDs in format: path::symbol_name
- Be concise but thorough
- Include practical examples
- Analyze time/space complexity when relevant
- Note dependencies on other symbols
- Output ONLY the JSON, no markdown, no explanations"#;

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

        // Call vLLM
        let response = self.client.generate(prompt).await?;
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

    /// Build the prompt for the LLM
    fn build_prompt(&self, file_ast: &FileAst, source_code: &str) -> String {
        // Serialize AST to JSON for context
        let ast_json =
            serde_json::to_string_pretty(&file_ast.ast).unwrap_or_else(|_| "{}".to_string());

        format!(
            "{}\n\n---\n\nFile: {}\nLanguage: {}\n\nSource Code:\n```\n{}\n```\n\nAST:\n```json\n{}\n```\n\nGenerate documentation:",
            SYSTEM_PROMPT,
            file_ast.file_path,
            file_ast.language,
            source_code,
            ast_json
        )
    }

    /// Parse the LLM's JSON response
    fn parse_llm_response(&self, text: &str) -> Result<Vec<Symbol>> {
        // Try to extract JSON from markdown code blocks if present
        let json_text = if text.contains("```json") {
            text.split("```json")
                .nth(1)
                .and_then(|s| s.split("```").next())
                .unwrap_or(text)
        } else if text.contains("```") {
            text.split("```")
                .nth(1)
                .and_then(|s| s.split("```").next())
                .unwrap_or(text)
        } else {
            text
        }
        .trim();

        // Parse JSON
        let response: LlmResponse = serde_json::from_str(json_text)?;
        Ok(response.symbols)
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
            log_info(&format!(
                "[{}/{}] Processing: {}",
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

/// Log an error message to stderr
fn log_error(message: &str) {
    eprintln!("[ERROR] {}", message);
    let _ = io::stderr().flush();
}
