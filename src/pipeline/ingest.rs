// ingest.rs
// Phase 1: Extract source files into memory from zip/git

use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;
use std::process::Command;
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
        InputSource::GitUrl(url) => {
            println!("Cloning git repo: {}", url);

            // Create a temporary directory inside the system temp dir
            let tmp_root = std::env::temp_dir();
            let unique = format!("doctown_clone_{}", chrono::Utc::now().timestamp_millis());
            let tmp_dir = tmp_root.join(unique);

            fs::create_dir_all(&tmp_dir)?;

            // Clone shallow to save time
            let status = Command::new("git")
                .arg("clone")
                .arg("--depth")
                .arg("1")
                .arg(&url)
                .arg(&tmp_dir)
                .status()
                .map_err(|e| format!("Failed to spawn git: {}", e))?;

            if !status.success() {
                // Clean up if clone failed
                let _ = fs::remove_dir_all(&tmp_dir);
                return Err(format!("Git clone failed for {}", url).into());
            }

            // Walk the cloned directory recursively and collect files
            let mut files = HashMap::new();

            fn visit_dir(
                base: &Path,
                dir: &Path,
                files: &mut HashMap<String, Vec<u8>>,
            ) -> Result<(), Box<dyn std::error::Error>> {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    let metadata = entry.metadata()?;

                    if metadata.is_dir() {
                        // Skip .git directory quickly
                        if path.file_name().and_then(|s| s.to_str()) == Some(".git") {
                            continue;
                        }
                        visit_dir(base, &path, files)?;
                    } else if metadata.is_file() {
                        if let Ok(rel) = path.strip_prefix(base) {
                            let rel_str = rel.to_string_lossy().replace("\\", "/");
                            if should_process(&rel_str) {
                                let bytes = fs::read(&path)?;
                                files.insert(rel_str, bytes);
                            }
                        }
                    }
                }
                Ok(())
            }

            visit_dir(&tmp_dir, &tmp_dir, &mut files)?;

            // Clean up temp directory
            let _ = fs::remove_dir_all(&tmp_dir);

            println!("Extracted {} files from git repo", files.len());
            Ok(files)
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

    // If filename matches a well-known extensionless file, accept it
    let basename = filename.split('/').last().unwrap_or("").to_lowercase();

    let extensionless_whitelist = [
        "readme",
        "license",
        "makefile",
        "cargo.toml",
        "package.json",
    ];
    if extensionless_whitelist.iter().any(|n| basename == *n) {
        return true;
    }

    extensions
        .iter()
        .any(|ext| filename.to_lowercase().ends_with(ext))
}
