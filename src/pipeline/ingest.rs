// ingest.rs
// Phase 1: Extract source files into memory from zip/git

use std::collections::HashMap;
use std::io::{Cursor, Read};
use zip::ZipArchive;

#[allow(dead_code)]
#[derive(Debug)]
pub enum InputSource {
    ZipBytes(Vec<u8>),
    ZipPath(String),
    GitUrl(String),
}

pub fn ingest(source: InputSource) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
    match source {
        InputSource::ZipBytes(bytes) => {
            println!("Processing zip from memory ({} bytes)", bytes.len());
            extract_zip_to_memory(bytes)
        }
        InputSource::ZipPath(path) => {
            println!("Reading zip from: {}", path);
            let bytes = std::fs::read(&path)?;
            extract_zip_to_memory(bytes)
        }
        InputSource::GitUrl(_url) => {
            // TODO: implement with git2 crate
            Err("Git URL support not yet implemented".into())
        }
    }
}

fn extract_zip_to_memory(
    zip_bytes: Vec<u8>,
) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
    let cursor = Cursor::new(zip_bytes);
    let mut archive = ZipArchive::new(cursor)?;
    let mut files = HashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;

        // Skip directories
        if file.is_dir() {
            continue;
        }

        let name = file.name().to_string();

        // Only process source code and relevant files
        if should_process(&name) {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            files.insert(name, buffer);
        }
    }

    println!("Extracted {} files to memory", files.len());
    Ok(files)
}

fn should_process(filename: &str) -> bool {
    // Skip common non-source directories
    let skip_dirs = [
        "node_modules/",
        ".git/",
        "target/",
        "build/",
        "dist/",
        "__pycache__/",
        ".venv/",
        "venv/",
        ".idea/",
        ".vscode/",
    ];

    if skip_dirs.iter().any(|dir| filename.contains(dir)) {
        return false;
    }

    // Process source code files
    let extensions = [
        ".rs",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".lua",
        ".sh",
        ".bash",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".md",
        ".html",
        ".css",
        ".swift",
        ".dockerfile",
    ];

    extensions
        .iter()
        .any(|ext| filename.to_lowercase().ends_with(ext))
}
