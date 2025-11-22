mod graph;
mod pipeline;
mod serverless;

use clap::Parser;
use graph::builder::build_graph;
use pipeline::ingest::{ingest, InputSource};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "doctown-builder")]
#[command(about = "Generate documentation packages from source code")]
struct Args {
    /// Input source (zip file path or git URL)
    #[arg(index = 1)]
    input: Option<String>,

    /// Run in serverless mode (for RunPod deployment)
    #[arg(long)]
    serverless: bool,

    /// GitHub token for private repos
    #[arg(long)]
    github_token: Option<String>,

    /// Git branch/tag/commit to checkout
    #[arg(long)]
    git_ref: Option<String>,

    /// Output directory
    #[arg(short, long, default_value = "output")]
    output: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();

    let args = Args::parse();

    if args.serverless {
        run_serverless().await
    } else {
        run_cli(args).await
    }
}

/// Run in serverless mode for RunPod
async fn run_serverless() -> Result<(), Box<dyn std::error::Error>> {
    use serverless::*;

    // Read job input
    let job = match read_job_input() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    };

    log(
        &job.job_id,
        "info",
        &format!("Starting build for {}", job.repo),
    );

    // Load serverless configuration
    #[allow(unused)]
    let config = match ServerlessConfig::from_env() {
        Ok(c) => c,
        Err(e) => {
            log(&job.job_id, "error", &format!("Configuration error: {}", e));
            let output = JobOutput {
                success: false,
                job_id: job.job_id.clone(),
                file_url: None,
                error: Some(format!("Configuration error: {}", e)),
                symbols_count: 0,
                files_count: 0,
            };
            // Try to send failure webhook even if config is partial
            eprintln!("ERROR: {}", e);
            std::process::exit(1);
        }
    };

    // Build the input source
    let source = if let Some(token) = &job.github_token {
        // Inject token into URL for private repos
        let authed_url = inject_github_token(&job.repo, token);
        InputSource::GitUrl(authed_url)
    } else {
        InputSource::GitUrl(job.repo.clone())
    };

    // Run the pipeline
    let result = run_pipeline(source, "output").await;

    match result {
        Ok((docpack_path, symbols_count, files_count)) => {
            log(&job.job_id, "info", "Pipeline complete, uploading to R2...");

            // Create R2 client and upload
            let r2_client = create_r2_client(&config).await?;
            let r2_key = generate_r2_key(&job.user_id, &job.repo);

            match upload_to_r2(
                &r2_client,
                &config.r2_bucket,
                Path::new(&docpack_path),
                &r2_key,
            )
            .await
            {
                Ok(file_url) => {
                    log(&job.job_id, "info", &format!("Uploaded to {}", file_url));

                    let output = JobOutput {
                        success: true,
                        job_id: job.job_id.clone(),
                        file_url: Some(file_url),
                        error: None,
                        symbols_count,
                        files_count,
                    };

                    if let Err(e) = send_webhook(&config, &output).await {
                        log(&job.job_id, "error", &format!("Webhook failed: {}", e));
                    } else {
                        log(&job.job_id, "info", "Webhook sent successfully");
                    }

                    // Output success for RunPod
                    println!("{}", serde_json::to_string(&output)?);
                }
                Err(e) => {
                    let output = JobOutput {
                        success: false,
                        job_id: job.job_id.clone(),
                        file_url: None,
                        error: Some(format!("R2 upload failed: {}", e)),
                        symbols_count,
                        files_count,
                    };
                    let _ = send_webhook(&config, &output).await;
                    eprintln!("ERROR: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            log(&job.job_id, "error", &format!("Pipeline failed: {}", e));

            let output = JobOutput {
                success: false,
                job_id: job.job_id.clone(),
                file_url: None,
                error: Some(e.to_string()),
                symbols_count: 0,
                files_count: 0,
            };

            let _ = send_webhook(&config, &output).await;

            // Output error for RunPod
            println!("{}", serde_json::to_string(&output)?);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Inject GitHub token into URL for authentication
fn inject_github_token(url: &str, token: &str) -> String {
    if url.starts_with("https://github.com/") {
        url.replace(
            "https://github.com/",
            &format!("https://{}@github.com/", token),
        )
    } else {
        url.to_string()
    }
}

/// Run in CLI mode
async fn run_cli(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    println!("Doctown Builder\n");

    let input = match args.input {
        Some(i) => i,
        None => {
            eprintln!("Usage: doctown-builder <path-to-zip|git-url>");
            eprintln!("\nExamples:");
            eprintln!("  doctown-builder myproject.zip");
            eprintln!("  doctown-builder https://github.com/user/repo.git");
            eprintln!("\nFlags:");
            eprintln!("  --serverless    Run in serverless mode for RunPod");
            eprintln!("  --github-token  GitHub token for private repos");
            eprintln!("  --git-ref       Branch/tag/commit to checkout");
            eprintln!("  -o, --output    Output directory (default: output)");
            std::process::exit(1);
        }
    };

    // Determine input type
    let source = if input.starts_with("http://") || input.starts_with("https://") {
        if input.contains("github.com") || input.ends_with(".git") {
            let url = if let Some(token) = args.github_token {
                inject_github_token(&input, &token)
            } else {
                input.clone()
            };
            InputSource::GitUrl(url)
        } else {
            return Err(format!("Unsupported URL: {}. Only Git URLs are supported.", input).into());
        }
    } else if input.ends_with(".zip") {
        InputSource::ZipPath(input.clone())
    } else {
        return Err(format!(
            "Unsupported file type: {}. Expected .zip file or Git URL.",
            input
        )
        .into());
    };

    println!("Input: {}", input);

    let (docpack_path, symbols_count, files_count) = run_pipeline(source, &args.output).await?;

    println!("\n✅ Build complete!");
    println!("   📦 Output: {}", docpack_path);
    println!("   📊 Symbols: {}", symbols_count);
    println!("   📄 Files: {}", files_count);

    Ok(())
}

/// Run the main pipeline, returning (output_path, symbols_count, files_count)
async fn run_pipeline(
    source: InputSource,
    output_dir: &str,
) -> Result<(String, usize, usize), Box<dyn std::error::Error>> {
    println!("\n🟪 Phase 1: Ingesting source files...");
    let files = ingest(source)?;
    let files_count = files.len();

    println!("Loaded {} files", files_count);

    println!("\n🟩 Phase 2: Parsing ASTs...");
    let parsed = pipeline::parse::parse_all_files(&files)?;

    println!("\n🟫 Phase 3: Building Docpack Graph...");
    let mut graph = build_graph(&parsed);

    println!("\n🟧 Phase 4: Analyzing metrics...");
    let analysis_config = pipeline::analyze::AnalysisConfig::default();
    let analysis_result = pipeline::analyze::analyze_graph(&mut graph, &analysis_config)?;

    println!("   Nodes analyzed: {}", analysis_result.nodes_analyzed);
    println!(
        "   Complexity calculated: {}",
        analysis_result.complexity_calculated
    );

    let stats = graph.stats();
    let symbols_count = stats.total_nodes;

    println!("\n🟨 Phase 5: Semantic Clustering...");
    let cluster_config = pipeline::cluster::ClusterConfig::default();
    let _cluster_result = pipeline::cluster::cluster_graph(&mut graph, &cluster_config)?;

    // Ensure output directory exists
    let output_path = Path::new(output_dir);
    if !output_path.exists() {
        std::fs::create_dir_all(output_path)?;
    }

    // Save graph to temporary location
    let temp_graph_path = format!("{}/.temp-graph.json", output_dir);
    println!("\n💾 Preparing graph data...");
    graph.save_to_file(&temp_graph_path)?;

    println!("\n🟥 Phase 6: Generating documentation with LLM...");
    let gen_config = pipeline::generate::GenerationConfig::default();

    match pipeline::generate::generate_documentation(&graph, &gen_config).await {
        Ok(result) => {
            println!("   Symbol summaries: {}", result.symbol_summaries.len());
            println!("   Module overviews: {}", result.module_overviews.len());
            println!("   Total tokens used: {}", result.total_tokens_used);

            // Save documentation
            let temp_doc_path = format!("{}/.temp-documentation.json", output_dir);
            pipeline::generate::save_documentation(&result, &temp_doc_path)?;

            // Package
            println!("\n📦 Phase 7: Packaging outputs...");
            let package_config = pipeline::package::PackageConfig::default();
            let package_result =
                pipeline::package::package_outputs("repo", output_dir, &package_config)?;

            Ok((package_result.output_path, symbols_count, files_count))
        }
        Err(e) => {
            eprintln!("Warning: LLM documentation failed: {}", e);
            eprintln!("Continuing with graph-only output...");

            // Package without documentation
            println!("\n📦 Phase 7: Packaging outputs...");
            let package_config = pipeline::package::PackageConfig::default();
            let package_result =
                pipeline::package::package_outputs("repo", output_dir, &package_config)?;

            Ok((package_result.output_path, symbols_count, files_count))
        }
    }
}
