use anyhow::{Context, Result};
use serde::Serialize;
use std::{collections::HashMap, fs::File, io::Write, path::Path};
use tree_sitter::{Language, Node, Parser};
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

fn main() -> Result<()> {
    let zip_path = "test-sources/xandwr.ca-main.zip";

    // Create a temporary directory that will be automatically cleaned up
    let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;
    let temp_path = temp_dir.path();

    // Extract zip to temporary directory
    ingest_zip(zip_path, temp_path)?;

    // Create parser pool for reuse
    let mut parser_pool = ParserPool::new()?;

    // Parse all supported files in the extracted directory with streaming output
    parse_directory_streaming(temp_path, &mut parser_pool, &mut std::io::stdout())?;

    // temp_dir is automatically cleaned up when it goes out of scope
    Ok(())
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

fn parse_directory_streaming<W: Write>(
    dir: &Path,
    parser_pool: &mut ParserPool,
    writer: &mut W,
) -> Result<()> {
    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();

        // Skip files larger than 512KB
        let size = match std::fs::metadata(path) {
            Ok(metadata) => metadata.len(),
            Err(_) => continue,
        };
        if size > MAX_FILE_SIZE {
            continue;
        }

        // Determine language from file extension
        let language = match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some("js") | Some("jsx") => "javascript",
            Some("ts") | Some("tsx") => "typescript",
            Some("go") => "go",
            _ => continue, // Skip unsupported file types
        };

        // Parse the file using parser pool
        if let Ok(ast) = parse_file_with_pool(path, language, parser_pool) {
            // Get relative path from temp directory for cleaner output
            let relative_path = path
                .strip_prefix(dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            let file_ast = FileAst {
                file_path: relative_path,
                language: language.to_string(),
                ast,
            };

            // Stream output immediately
            let json = serde_json::to_string(&file_ast)?;
            writeln!(writer, "{}", json)?;
        }
    }

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
