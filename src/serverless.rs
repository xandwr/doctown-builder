//! Serverless handler for RunPod deployment
//!
//! This module provides functionality to run the builder in serverless mode,
//! accepting job configuration via stdin JSON and uploading results to R2.

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::primitives::ByteStream;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;

/// Job input from RunPod
#[derive(Debug, Deserialize)]
pub struct JobInput {
    pub job_id: String,
    pub user_id: String,
    pub repo: String,
    pub git_ref: String,
    pub github_token: Option<String>,
}

/// Job output sent to webhook
#[derive(Debug, Serialize)]
pub struct JobOutput {
    pub success: bool,
    pub job_id: String,
    pub file_url: Option<String>,
    pub error: Option<String>,
    pub symbols_count: usize,
    pub files_count: usize,
}

/// Progress log entry
#[derive(Debug, Serialize)]
pub struct LogEntry {
    pub job_id: String,
    pub level: String,
    pub message: String,
    pub timestamp: String,
}

/// Configuration for serverless mode
pub struct ServerlessConfig {
    pub r2_endpoint: String,
    pub r2_bucket: String,
    pub r2_access_key: String,
    pub r2_secret_key: String,
    pub webhook_url: String,
    pub webhook_secret: String,
}

impl ServerlessConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, String> {
        Ok(Self {
            r2_endpoint: env::var("BUCKET_S3_ENDPOINT")
                .map_err(|_| "Missing BUCKET_S3_ENDPOINT")?,
            r2_bucket: env::var("BUCKET_NAME").unwrap_or_else(|_| "doctown-central".to_string()),
            r2_access_key: env::var("BUCKET_ACCESS_KEY_ID")
                .map_err(|_| "Missing BUCKET_ACCESS_KEY_ID")?,
            r2_secret_key: env::var("BUCKET_SECRET_ACCESS_KEY")
                .map_err(|_| "Missing BUCKET_SECRET_ACCESS_KEY")?,
            webhook_url: env::var("WEBHOOK_URL")
                .unwrap_or_else(|_| "https://doctown.dev/api/jobs/complete".to_string()),
            webhook_secret: env::var("DOCTOWN_BUILDER_SHARED_SECRET")
                .map_err(|_| "Missing DOCTOWN_BUILDER_SHARED_SECRET")?,
        })
    }
}

/// Log a message (sends to stdout for RunPod to capture)
pub fn log(job_id: &str, level: &str, message: &str) {
    let entry = LogEntry {
        job_id: job_id.to_string(),
        level: level.to_string(),
        message: message.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    println!("{}", serde_json::to_string(&entry).unwrap_or_default());
}

/// Create S3 client configured for Cloudflare R2
pub async fn create_r2_client(config: &ServerlessConfig) -> Result<S3Client, String> {
    let creds = aws_sdk_s3::config::Credentials::new(
        &config.r2_access_key,
        &config.r2_secret_key,
        None,
        None,
        "r2",
    );

    let s3_config = aws_sdk_s3::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .endpoint_url(&config.r2_endpoint)
        .region(aws_sdk_s3::config::Region::new("auto"))
        .credentials_provider(creds)
        .force_path_style(true)
        .build();

    Ok(S3Client::from_conf(s3_config))
}

/// Upload a docpack file to R2
pub async fn upload_to_r2(
    client: &S3Client,
    bucket: &str,
    file_path: &Path,
    key: &str,
) -> Result<String, String> {
    let body = fs::read(file_path).map_err(|e| format!("Failed to read file: {}", e))?;

    let byte_stream = ByteStream::from(body);

    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(byte_stream)
        .content_type("application/zip")
        .send()
        .await
        .map_err(|e| format!("Failed to upload to R2: {}", e))?;

    // Return the public URL
    Ok(format!(
        "{}/{}/{}",
        "https://storage.doctown.dev", bucket, key
    ))
}

/// Send completion webhook to the doctown API
pub async fn send_webhook(config: &ServerlessConfig, output: &JobOutput) -> Result<(), String> {
    let client = reqwest::Client::new();

    let response = client
        .post(&config.webhook_url)
        .header("Authorization", format!("Bearer {}", config.webhook_secret))
        .header("Content-Type", "application/json")
        .json(output)
        .send()
        .await
        .map_err(|e| format!("Failed to send webhook: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Webhook failed with status {}: {}", status, body));
    }

    Ok(())
}

/// Generate the R2 key for a docpack
pub fn generate_r2_key(user_id: &str, repo: &str) -> String {
    // Extract repo name from URL
    let repo_name = repo
        .trim_end_matches(".git")
        .split('/')
        .last()
        .unwrap_or("unknown");

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    format!("docpacks/{}/{}-{}.docpack", user_id, repo_name, timestamp)
}

/// Read job input from stdin (RunPod passes input this way)
pub fn read_job_input() -> Result<JobInput, String> {
    // RunPod passes the input as an environment variable
    let input_json = env::var("RUNPOD_INPUT")
        .or_else(|_| {
            // Fallback: read from stdin
            use std::io::Read;
            let mut input = String::new();
            std::io::stdin()
                .read_to_string(&mut input)
                .map(|_| input)
                .map_err(|e| e.to_string())
        })
        .map_err(|_| "No input provided. Set RUNPOD_INPUT or pipe JSON to stdin.")?;

    serde_json::from_str(&input_json).map_err(|e| format!("Invalid job input JSON: {}", e))
}
