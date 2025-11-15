use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Docpack format version
pub const DOCPACK_FORMAT_VERSION: u32 = 1;

/// Generator metadata
pub const GENERATOR_NAME: &str = "doctown-generator";
pub const GENERATOR_VERSION: &str = "0.1.0";

// ============================================================================
// Manifest structures
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub docpack_format: u32,
    pub name: String,
    pub repo: String,
    pub commit: String,
    pub generated_at: DateTime<Utc>,
    pub version: String,
    pub generator: GeneratorInfo,
    pub stats: Stats,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeneratorInfo {
    pub name: String,
    pub version: String,
    pub model: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Stats {
    pub files_total: usize,
    pub symbols_total: usize,
    pub tokens_input: u64,
    pub tokens_output: u64,
}

// ============================================================================
// Tree structures (file metadata)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct TreeEntry {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loc: Option<usize>,
    pub kind: EntryKind,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryKind {
    File,
    Directory,
}

impl TreeEntry {
    pub fn file(path: String, language: String, content: &[u8], loc: usize) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let sha256 = hex::encode(hasher.finalize());

        Self {
            path,
            language: Some(language),
            sha256: Some(sha256),
            loc: Some(loc),
            kind: EntryKind::File,
        }
    }

    pub fn directory(path: String) -> Self {
        Self {
            path,
            language: None,
            sha256: None,
            loc: None,
            kind: EntryKind::Directory,
        }
    }
}

// ============================================================================
// Content structures (AI-generated documentation)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentEntry {
    pub path: String,
    pub symbols: Vec<Symbol>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Symbol {
    pub id: String,
    pub kind: SymbolKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    pub summary: String,
    pub description: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub complexity: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    Constant,
    Variable,
    Module,
    Type,
}

// ============================================================================
// Docpack builder - writes the final .docpack (zip) file
// ============================================================================

pub struct DocpackBuilder {
    output_path: PathBuf,
    temp_dir: PathBuf,
    manifest_file: BufWriter<File>,
    tree_file: BufWriter<File>,
    content_file: BufWriter<File>,
}

impl DocpackBuilder {
    pub fn new<P: AsRef<Path>>(output_path: P) -> Result<Self> {
        let output_path = output_path.as_ref().to_path_buf();

        // Create temp directory for building docpack contents
        let temp_dir = tempfile::tempdir()?.path().to_path_buf();

        // Create files
        let manifest_path = temp_dir.join("manifest.json");
        let tree_path = temp_dir.join("tree.jsonl");
        let content_path = temp_dir.join("content.jsonl");

        let manifest_file = BufWriter::new(File::create(&manifest_path)?);
        let tree_file = BufWriter::new(File::create(&tree_path)?);
        let content_file = BufWriter::new(File::create(&content_path)?);

        // Create META directory
        fs::create_dir(temp_dir.join("META"))?;

        Ok(Self {
            output_path,
            temp_dir,
            manifest_file,
            tree_file,
            content_file,
        })
    }

    pub fn write_manifest(&mut self, manifest: &Manifest) -> Result<()> {
        let json = serde_json::to_string_pretty(manifest)?;
        writeln!(self.manifest_file, "{}", json)?;
        self.manifest_file.flush()?;
        Ok(())
    }

    pub fn write_tree_entry(&mut self, entry: &TreeEntry) -> Result<()> {
        let json = serde_json::to_string(entry)?;
        writeln!(self.tree_file, "{}", json)?;
        Ok(())
    }

    pub fn write_content_entry(&mut self, entry: &ContentEntry) -> Result<()> {
        let json = serde_json::to_string(entry)?;
        writeln!(self.content_file, "{}", json)?;
        Ok(())
    }

    pub fn write_meta_file<P: AsRef<Path>>(&self, filename: P, content: &[u8]) -> Result<()> {
        let path = self.temp_dir.join("META").join(filename);
        fs::write(path, content)?;
        Ok(())
    }

    /// Finalize and create the .docpack zip file
    pub fn finalize(mut self) -> Result<PathBuf> {
        // Flush all writers
        self.manifest_file.flush()?;
        self.tree_file.flush()?;
        self.content_file.flush()?;

        // Drop writers to close files
        drop(self.manifest_file);
        drop(self.tree_file);
        drop(self.content_file);

        // Create zip archive
        let zip_file = File::create(&self.output_path)?;
        let mut zip = zip::ZipWriter::new(zip_file);

        let options = zip::write::FileOptions::<()>::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // Add all files from temp directory to zip
        for entry in walkdir::WalkDir::new(&self.temp_dir) {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let relative_path = path.strip_prefix(&self.temp_dir)?;
                let name = relative_path.to_string_lossy();

                zip.start_file(name, options)?;
                let content = fs::read(path)?;
                zip.write_all(&content)?;
            }
        }

        zip.finish()?;

        // Clean up temp directory
        fs::remove_dir_all(&self.temp_dir)?;

        Ok(self.output_path.clone())
    }
}
