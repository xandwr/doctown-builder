use anyhow::{Context, Result};
use aws_config::meta::region::RegionProviderChain;
use aws_credential_types::Credentials;
use aws_sdk_s3::{config::Region, primitives::ByteStream, Client};
use std::env;
use std::path::Path;

/// S3/R2 client for uploading docpack files
pub struct S3Client {
    client: Client,
    bucket_name: String,
}

impl S3Client {
    /// Create a new S3Client from environment variables
    pub async fn from_env() -> Result<Self> {
        let access_key_id = env::var("BUCKET_ACCESS_KEY_ID")
            .context("BUCKET_ACCESS_KEY_ID not found in environment")?;
        let secret_access_key = env::var("BUCKET_SECRET_ACCESS_KEY")
            .context("BUCKET_SECRET_ACCESS_KEY not found in environment")?;
        let bucket_name = env::var("BUCKET_NAME").unwrap_or_else(|_| "doctown-central".to_string());
        let endpoint_url = env::var("BUCKET_ENDPOINT_URL")
            .context("BUCKET_ENDPOINT_URL not found in environment (e.g., https://your-account-id.r2.cloudflarestorage.com)")?;

        // Create credentials
        let credentials = Credentials::new(access_key_id, secret_access_key, None, None, "env");

        // Build S3 config for R2
        let region = RegionProviderChain::default_provider().or_else(Region::new("auto"));

        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(region)
            .credentials_provider(credentials)
            .endpoint_url(endpoint_url)
            .load()
            .await;

        let client = Client::new(&config);

        Ok(Self {
            client,
            bucket_name,
        })
    }

    /// Upload a docpack file to S3/R2
    ///
    /// # Arguments
    /// * `file_path` - Local path to the .docpack file
    /// * `owner` - GitHub repository owner (used for S3 key)
    /// * `repo_name` - GitHub repository name (used for S3 key)
    ///
    /// # Returns
    /// The S3 key where the file was uploaded (e.g., "docpacks/owner/repo.docpack")
    pub async fn upload_docpack<P: AsRef<Path>>(
        &self,
        file_path: P,
        owner: &str,
        repo_name: &str,
    ) -> Result<String> {
        let file_path = file_path.as_ref();

        // Construct S3 key: docpacks/{owner}/{repo_name}.docpack
        let s3_key = format!("docpacks/{}/{}.docpack", owner, repo_name);

        // Read file as ByteStream
        let body = ByteStream::from_path(&file_path)
            .await
            .context(format!("Failed to read file: {}", file_path.display()))?;

        // Upload to S3/R2
        self.client
            .put_object()
            .bucket(&self.bucket_name)
            .key(&s3_key)
            .body(body)
            .content_type("application/zip")
            .send()
            .await
            .context(format!("Failed to upload to S3: {}", s3_key))?;

        log_info(&format!(
            "✓ Uploaded to S3: {}/{}",
            self.bucket_name, s3_key
        ));

        Ok(s3_key)
    }

    /// Get the public URL for a docpack
    ///
    /// Note: This assumes the bucket is configured for public read access.
    /// For Cloudflare R2, you'll need to set up a public domain or use R2's public buckets feature.
    pub fn get_public_url(&self, s3_key: &str) -> String {
        // For R2 with public access, the URL format is typically:
        // https://pub-{bucket-id}.r2.dev/{key}
        // Or with custom domain: https://your-domain.com/{key}

        // For now, return the S3 URI - you'll need to configure this based on your R2 setup
        format!("s3://{}/{}", self.bucket_name, s3_key)
    }
}

/// Log an informational message to stderr (won't interfere with stdout)
fn log_info(message: &str) {
    eprintln!("[LOG] {}", message);
    use std::io::{self, Write};
    let _ = io::stderr().flush();
}
