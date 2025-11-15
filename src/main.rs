mod docpack;
mod generator;
mod s3;
mod vllm;

use anyhow::{Context, Result};
use chrono::Utc;
use docpack::{
    DocpackBuilder, GeneratorInfo, Manifest, Stats, TreeEntry, DOCPACK_FORMAT_VERSION,
    GENERATOR_NAME, GENERATOR_VERSION,
};
use generator::BatchProcessor;
use s3::S3Client;
use serde::Serialize;
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{self, Write},
    path::Path,
};
use tree_sitter::{Language, Node, Parser};
use vllm::VllmClient;
use walkdir::WalkDir;

// File size threshold: skip files larger than 512KB
const MAX_FILE_SIZE: u64 = 512 * 1024;

// Only emit AST nodes that can produce symbols
const INTERESTING_KINDS: &[&str] = &[
    "function_declaration",
    "function_item",
    "method_definition",
    "class_declaration",
    "struct_item",
    "enum_item",
    "trait_item",
    "impl_item",
    "module",
    "interface_declaration",
    "type_alias_declaration",
    "assignment_expression",
    "export_statement",
    "import_statement",
    "const_item",
    "static_item",
];

#[derive(Serialize)]
struct AstNode {
    kind: String,
    start_byte: usize,
    end_byte: usize,
    start_point: Point,
    end_point: Point,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    children: Vec<AstNode>,
}

#[derive(Serialize)]
struct Point {
    row: usize,
    column: usize,
}

#[derive(Serialize)]
struct FileAst {
    file_path: String,
    language: String,
    ast: AstNode,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenv::dotenv().ok();

    // Get command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        log_error("Usage: doctown-builder <zip-file> <repo-url> <commit-hash> <output-docpack>");
        log_error(
            "Example: doctown-builder repo.zip https://github.com/user/repo abc123 output.docpack",
        );
        anyhow::bail!("Missing required arguments");
    }

    let zip_path = &args[1];
    let repo_url = &args[2];
    let commit_hash = &args[3];
    let output_path = &args[4];

    log_info(&format!("Starting doctown-builder"));
    log_info(&format!("  Repo: {}", repo_url));
    log_info(&format!("  Commit: {}", commit_hash));
    log_info(&format!("  Output: {}", output_path));

    // Step 1: Extract zip to temporary directory
    log_info("Step 1: Extracting ZIP file...");
    let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;
    let temp_path = temp_dir.path();
    ingest_zip(zip_path, temp_path)?;
    log_info("✓ ZIP extraction complete");

    // Step 2: Parse all files and collect AST + source code
    log_info("Step 2: Parsing source files...");
    let mut parser_pool = ParserPool::new()?;
    let parsed_files = parse_directory_collect(temp_path, &mut parser_pool)?;
    log_info(&format!("✓ Parsed {} files", parsed_files.len()));

    // Step 3: Initialize vLLM client
    log_info("Step 3: Connecting to vLLM endpoint...");
    let vllm_client = VllmClient::from_env()?;
    log_info("✓ vLLM client initialized");

    // Step 4: Generate documentation using LLM
    log_info("Step 4: Generating documentation with AI...");
    let mut batch_processor = BatchProcessor::new(vllm_client, 1);
    let content_entries = batch_processor.process_files(parsed_files).await?;
    let (tokens_in, tokens_out) = batch_processor.get_token_stats();
    log_info(&format!(
        "✓ Generated docs ({} input tokens, {} output tokens)",
        tokens_in, tokens_out
    ));

    // Step 5: Build docpack
    log_info("Step 5: Building .docpack file...");

    // Re-parse to get file list again (since we moved parsed_files)
    let mut parser_pool2 = ParserPool::new()?;
    let parsed_files2 = parse_directory_collect(temp_path, &mut parser_pool2)?;

    let mut docpack_builder = DocpackBuilder::new(output_path)?;

    // Write manifest
    let symbols_total: usize = content_entries.iter().map(|e| e.symbols.len()).sum();
    let manifest = Manifest {
        docpack_format: DOCPACK_FORMAT_VERSION,
        name: extract_repo_name(repo_url),
        repo: repo_url.to_string(),
        commit: commit_hash.to_string(),
        generated_at: Utc::now(),
        version: commit_hash.to_string(),
        generator: GeneratorInfo {
            name: GENERATOR_NAME.to_string(),
            version: GENERATOR_VERSION.to_string(),
            model: "qwen2.5-coder-14b-instruct".to_string(),
        },
        stats: Stats {
            files_total: parsed_files2.len(),
            symbols_total,
            tokens_input: tokens_in,
            tokens_output: tokens_out,
        },
    };
    docpack_builder.write_manifest(&manifest)?;

    // Write tree entries (file metadata)
    for (file_ast, source_code) in &parsed_files2 {
        let loc = source_code.lines().count();
        let entry = TreeEntry::file(
            file_ast.file_path.clone(),
            file_ast.language.clone(),
            source_code.as_bytes(),
            loc,
        );
        docpack_builder.write_tree_entry(&entry)?;
    }

    // Write content entries (AI-generated docs)
    for content in content_entries {
        docpack_builder.write_content_entry(&content)?;
    }

    // Finalize and create zip
    let final_path = docpack_builder.finalize()?;
    log_info(&format!("✓ Docpack created: {}", final_path.display()));

    // Step 6: Upload to S3/R2
    log_info("Step 6: Uploading to R2...");
    let s3_client = S3Client::from_env().await?;

    // Extract owner and repo name from repo_url
    let (owner, repo_name) = extract_owner_and_repo(repo_url)?;

    let s3_key = s3_client
        .upload_docpack(&final_path, &owner, &repo_name)
        .await?;

    log_info("✅ Pipeline complete!");

    // Output S3 key to stdout so handler.py can read it
    println!("{}", s3_key);

    Ok(())
}

fn extract_repo_name(repo_url: &str) -> String {
    repo_url
        .trim_end_matches(".git")
        .split('/')
        .last()
        .unwrap_or("unknown")
        .to_string()
}

/// Extract owner and repo name from GitHub URL
/// Example: "https://github.com/owner/repo" -> ("owner", "repo")
fn extract_owner_and_repo(repo_url: &str) -> Result<(String, String)> {
    let url = repo_url.trim_end_matches(".git");
    let parts: Vec<&str> = url.split('/').collect();

    if parts.len() < 2 {
        anyhow::bail!("Invalid GitHub URL format: {}", repo_url);
    }

    let owner = parts[parts.len() - 2].to_string();
    let repo_name = parts[parts.len() - 1].to_string();

    Ok((owner, repo_name))
}

/// Log an informational message to stderr (won't interfere with stdout JSONL)
fn log_info(message: &str) {
    eprintln!("[LOG] {}", message);
    let _ = io::stderr().flush();
}

/// Log an error message to stderr
fn log_error(message: &str) {
    eprintln!("[ERROR] {}", message);
    let _ = io::stderr().flush();
}

/// Parser pool to reuse parsers across files
struct ParserPool {
    parsers: HashMap<String, Parser>,
}

impl ParserPool {
    fn new() -> Result<Self> {
        Ok(Self {
            parsers: HashMap::new(),
        })
    }

    fn get_parser(&mut self, language: &str) -> Result<&mut Parser> {
        if !self.parsers.contains_key(language) {
            let mut parser = Parser::new();
            let lang = Self::get_language(language)?;
            parser
                .set_language(&lang)
                .context("Failed to set parser language")?;
            self.parsers.insert(language.to_string(), parser);
        }
        Ok(self.parsers.get_mut(language).unwrap())
    }

    fn get_language(language: &str) -> Result<Language> {
        match language {
            "rust" => Ok(tree_sitter_rust::LANGUAGE.into()),
            "python" => Ok(tree_sitter_python::LANGUAGE.into()),
            "javascript" => Ok(tree_sitter_javascript::LANGUAGE.into()),
            "typescript" => Ok(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            "go" => Ok(tree_sitter_go::LANGUAGE.into()),
            "java" => Ok(tree_sitter_java::LANGUAGE.into()),
            "c" => Ok(tree_sitter_c::LANGUAGE.into()),
            "cpp" => Ok(tree_sitter_cpp::LANGUAGE.into()),
            "csharp" => Ok(tree_sitter_c_sharp::LANGUAGE.into()),
            "ruby" => Ok(tree_sitter_ruby::LANGUAGE.into()),
            "php" => Ok(tree_sitter_php::LANGUAGE_PHP.into()),
            "swift" => Ok(tree_sitter_swift::LANGUAGE.into()),
            "scala" => Ok(tree_sitter_scala::LANGUAGE.into()),
            "html" => Ok(tree_sitter_html::LANGUAGE.into()),
            "css" => Ok(tree_sitter_css::LANGUAGE.into()),
            "json" => Ok(tree_sitter_json::LANGUAGE.into()),
            "yaml" => Ok(tree_sitter_yaml::LANGUAGE.into()),
            "bash" => Ok(tree_sitter_bash::LANGUAGE.into()),
            _ => anyhow::bail!("Unsupported language: {}", language),
        }
    }
}

pub fn ingest_zip<Z: AsRef<Path>, O: AsRef<Path>>(zip_path: Z, out_dir: O) -> Result<()> {
    let file = File::open(zip_path).context("Failed to open zip file")?;
    let reader = std::io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(reader).context("Failed to read zip archive")?;
    std::fs::create_dir_all(&out_dir).context("Failed to create output directory")?;
    archive
        .extract(out_dir)
        .context("Failed to extract zip archive")?;
    Ok(())
}

fn parse_directory_collect(
    dir: &Path,
    parser_pool: &mut ParserPool,
) -> Result<Vec<(FileAst, String)>> {
    let mut results = Vec::new();

    // Collect all files
    let all_files: Vec<_> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    let total_files = all_files.len();
    log_info(&format!("Found {} files to scan", total_files));

    for entry in all_files {
        let path = entry.path();

        // Skip files larger than 512KB
        let size = match std::fs::metadata(path) {
            Ok(metadata) => metadata.len(),
            Err(_) => continue,
        };
        if size > MAX_FILE_SIZE {
            log_info(&format!(
                "Skipping large file: {} ({} bytes)",
                path.display(),
                size
            ));
            continue;
        }

        // Determine language from file extension
        let language = match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some("js") | Some("jsx") => "javascript",
            Some("ts") | Some("tsx") => "typescript",
            Some("go") => "go",
            Some("java") => "java",
            Some("c") | Some("h") => "c",
            Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hxx") => "cpp",
            Some("cs") => "csharp",
            Some("rb") => "ruby",
            Some("php") => "php",
            Some("swift") => "swift",
            Some("scala") | Some("sc") => "scala",
            Some("html") | Some("htm") => "html",
            Some("css") => "css",
            Some("json") => "json",
            Some("yaml") | Some("yml") => "yaml",
            Some("sh") | Some("bash") => "bash",
            _ => continue, // Skip unsupported file types
        };

        // Get relative path
        let relative_path = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Read source code
        let source_code = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                log_error(&format!("Failed to read {}: {}", relative_path, e));
                continue;
            }
        };

        // Parse the file
        match parse_file_with_pool(path, language, parser_pool) {
            Ok(ast) => {
                let file_ast = FileAst {
                    file_path: relative_path.clone(),
                    language: language.to_string(),
                    ast,
                };
                results.push((file_ast, source_code));
            }
            Err(e) => {
                log_error(&format!("Failed to parse {}: {}", relative_path, e));
            }
        }
    }

    log_info(&format!("Successfully parsed {} files", results.len()));
    Ok(results)
}

fn parse_directory_streaming<W: Write>(
    dir: &Path,
    parser_pool: &mut ParserPool,
    writer: &mut W,
) -> Result<()> {
    let mut _file_count = 0;
    let mut processed_count = 0;

    // Collect all files first to get total count
    let all_files: Vec<_> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    let total_files = all_files.len();
    log_info(&format!("Found {} files to scan", total_files));

    for entry in all_files {
        let path = entry.path();
        _file_count += 1;

        // Skip files larger than 512KB
        let size = match std::fs::metadata(path) {
            Ok(metadata) => metadata.len(),
            Err(_) => continue,
        };
        if size > MAX_FILE_SIZE {
            log_info(&format!(
                "Skipping large file: {} ({} bytes)",
                path.display(),
                size
            ));
            continue;
        }

        // Determine language from file extension
        let language = match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some("js") | Some("jsx") => "javascript",
            Some("ts") | Some("tsx") => "typescript",
            Some("go") => "go",
            Some("java") => "java",
            Some("c") | Some("h") => "c",
            Some("cpp") | Some("cc") | Some("cxx") | Some("hpp") | Some("hxx") => "cpp",
            Some("cs") => "csharp",
            Some("rb") => "ruby",
            Some("php") => "php",
            Some("swift") => "swift",
            Some("scala") | Some("sc") => "scala",
            Some("html") | Some("htm") => "html",
            Some("css") => "css",
            Some("json") => "json",
            Some("yaml") | Some("yml") => "yaml",
            Some("sh") | Some("bash") => "bash",
            _ => continue, // Skip unsupported file types
        };

        // Get relative path for display
        let relative_path = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        // Parse the file using parser pool
        match parse_file_with_pool(path, language, parser_pool) {
            Ok(ast) => {
                processed_count += 1;
                log_info(&format!(
                    "[{}/{}] Processing: {} ({})",
                    processed_count, total_files, relative_path, language
                ));

                let file_ast = FileAst {
                    file_path: relative_path,
                    language: language.to_string(),
                    ast,
                };

                // Stream output immediately
                let json = serde_json::to_string(&file_ast)?;
                writeln!(writer, "{}", json)?;
                writer.flush()?; // Ensure output is flushed for real-time streaming
            }
            Err(e) => {
                log_error(&format!("Failed to parse {}: {}", relative_path, e));
            }
        }
    }

    log_info(&format!(
        "Completed: processed {} files out of {} total files",
        processed_count, total_files
    ));

    Ok(())
}

fn parse_file_with_pool(
    path: &Path,
    language: &str,
    parser_pool: &mut ParserPool,
) -> Result<AstNode> {
    // Read file contents
    let source_code = std::fs::read_to_string(path)
        .context(format!("Failed to read file: {}", path.display()))?;

    // Get parser from pool
    let parser = parser_pool.get_parser(language)?;

    // Parse the source code
    let tree = parser
        .parse(&source_code, None)
        .context("Failed to parse source code")?;

    // Convert tree-sitter tree to our serializable format
    Ok(tree_to_ast(tree.root_node()))
}

fn tree_to_ast(node: Node) -> AstNode {
    let mut children = Vec::new();
    let mut cursor = node.walk();

    // Filter children to only include interesting kinds
    for child in node.children(&mut cursor) {
        let child_kind = child.kind();

        // Always recurse into named nodes and interesting nodes
        if child.is_named() && INTERESTING_KINDS.contains(&child_kind) {
            children.push(tree_to_ast(child));
        } else if child.is_named() {
            // For non-interesting named nodes, check if they have interesting descendants
            for grandchild in child.children(&mut child.walk()) {
                if grandchild.is_named() {
                    children.push(tree_to_ast(grandchild));
                }
            }
        }
    }

    AstNode {
        kind: node.kind().to_string(),
        start_byte: node.start_byte(),
        end_byte: node.end_byte(),
        start_point: Point {
            row: node.start_position().row,
            column: node.start_position().column,
        },
        end_point: Point {
            row: node.end_position().row,
            column: node.end_position().column,
        },
        text: None, // Don't store leaf text - extract later using byte ranges
        children,
    }
}
