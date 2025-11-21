// parse.rs
// Phase 2: Generate ASTs via tree-sitter

use std::collections::HashMap;
use tree_sitter::{Language, Parser, Tree};

#[derive(Debug)]
pub struct ParsedFile {
    pub filename: String,
    pub language: String,
    pub tree: Tree,
    pub source: Vec<u8>,
}

pub fn parse_all_files(
    files: &HashMap<String, Vec<u8>>,
) -> Result<Vec<ParsedFile>, Box<dyn std::error::Error>> {
    let mut parsed_files = Vec::new();
    let mut success_count = 0;
    let mut skip_count = 0;

    for (filename, content) in files {
        match parse_file(filename, content) {
            Ok(Some(parsed)) => {
                parsed_files.push(parsed);
                success_count += 1;
            }
            Ok(None) => {
                skip_count += 1;
            }
            Err(e) => {
                eprintln!("Failed to parse {}: {}", filename, e);
            }
        }
    }

    println!(
        "Parsed {} files ({} skipped)",
        success_count, skip_count
    );
    Ok(parsed_files)
}

fn parse_file(
    filename: &str,
    content: &[u8],
) -> Result<Option<ParsedFile>, Box<dyn std::error::Error>> {
    let (language_name, language) = match get_language_for_file(filename) {
        Some(lang) => lang,
        None => return Ok(None), // Skip unsupported files
    };

    let mut parser = Parser::new();
    parser.set_language(&language)?;

    match parser.parse(content, None) {
        Some(tree) => Ok(Some(ParsedFile {
            filename: filename.to_string(),
            language: language_name.to_string(),
            tree,
            source: content.to_vec(),
        })),
        None => Err(format!("Failed to parse {}", filename).into()),
    }
}

fn get_language_for_file(filename: &str) -> Option<(&'static str, Language)> {
    let lower = filename.to_lowercase();

    if lower.ends_with(".rs") {
        Some(("rust", tree_sitter_rust::LANGUAGE.into()))
    } else if lower.ends_with(".py") {
        Some(("python", tree_sitter_python::LANGUAGE.into()))
    } else if lower.ends_with(".js") || lower.ends_with(".mjs") || lower.ends_with(".cjs") {
        Some(("javascript", tree_sitter_javascript::LANGUAGE.into()))
    } else if lower.ends_with(".ts") {
        Some((
            "typescript",
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        ))
    } else if lower.ends_with(".tsx") {
        Some(("tsx", tree_sitter_typescript::LANGUAGE_TSX.into()))
    } else if lower.ends_with(".jsx") {
        Some(("jsx", tree_sitter_javascript::LANGUAGE.into()))
    } else if lower.ends_with(".go") {
        Some(("go", tree_sitter_go::LANGUAGE.into()))
    } else if lower.ends_with(".java") {
        Some(("java", tree_sitter_java::LANGUAGE.into()))
    } else if lower.ends_with(".c") || lower.ends_with(".h") {
        Some(("c", tree_sitter_c::LANGUAGE.into()))
    } else if lower.ends_with(".cpp")
        || lower.ends_with(".cc")
        || lower.ends_with(".cxx")
        || lower.ends_with(".hpp")
        || lower.ends_with(".hh")
    {
        Some(("cpp", tree_sitter_cpp::LANGUAGE.into()))
    } else if lower.ends_with(".cs") {
        Some(("c_sharp", tree_sitter_c_sharp::LANGUAGE.into()))
    } else if lower.ends_with(".rb") {
        Some(("ruby", tree_sitter_ruby::LANGUAGE.into()))
    } else if lower.ends_with(".lua") {
        Some(("lua", tree_sitter_lua::LANGUAGE.into()))
    } else if lower.ends_with(".sh") || lower.ends_with(".bash") {
        Some(("bash", tree_sitter_bash::LANGUAGE.into()))
    } else if lower.ends_with(".swift") {
        Some(("swift", tree_sitter_swift::LANGUAGE.into()))
    } else if lower.ends_with(".html") || lower.ends_with(".htm") {
        Some(("html", tree_sitter_html::LANGUAGE.into()))
    } else if lower.ends_with(".css") {
        Some(("css", tree_sitter_css::LANGUAGE.into()))
    } else if lower.ends_with(".json") {
        Some(("json", tree_sitter_json::LANGUAGE.into()))
    } else if lower.ends_with(".yaml") || lower.ends_with(".yml") {
        Some(("yaml", tree_sitter_yaml::LANGUAGE.into()))
    // Note: SQL, TOML, Markdown, Dockerfile have tree-sitter version mismatches
    // Skipping these for now - they'll be filtered out
    } else if lower.ends_with(".sql")
        || lower.ends_with(".toml")
        || lower.ends_with(".md")
        || lower.contains("dockerfile")
    {
        None
    } else if lower.ends_with(".ex") || lower.ends_with(".exs") {
        Some(("elixir", tree_sitter_elixir::LANGUAGE.into()))
    } else {
        None
    }
}
