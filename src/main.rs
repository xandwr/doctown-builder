mod graph;
mod pipeline;

use graph::builder::build_graph;
use pipeline::ingest::{InputSource, ingest};
use std::env;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    println!("Doctown Builder\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-zip|git-url>", args[0]);
        eprintln!("\nExamples:");
        eprintln!("  {} myproject.zip", args[0]);
        eprintln!("  {} https://github.com/user/repo.git", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];

    // Determine input type
    let source = if input.starts_with("http://") || input.starts_with("https://") {
        if input.contains("github.com") || input.ends_with(".git") {
            InputSource::GitUrl(input.clone())
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
    println!("\n🟪 Phase 1: Ingesting source files...");
    let files = ingest(source)?;

    println!("\nLoaded {} files:", files.len());
    for (name, content) in files.iter().take(10) {
        println!("   • {} ({} bytes)", name, content.len());
    }

    if files.len() > 10 {
        println!("   ... and {} more files", files.len() - 10);
    }

    println!("\n🟩 Phase 2: Parsing ASTs...");
    let parsed = pipeline::parse::parse_all_files(&files)?;

    // Show some sample AST output
    println!("\n📊 Sample parsed files:");
    for parsed_file in parsed.iter().take(5) {
        let root = parsed_file.tree.root_node();
        println!(
            "   • {} ({}): {} nodes, {} bytes",
            parsed_file.filename,
            parsed_file.language,
            root.descendant_count(),
            parsed_file.source.len()
        );

        // Show first few lines of S-expression (tree structure)
        let sexp = root.to_sexp();
        let preview: String = sexp.chars().take(120).collect();
        println!("     Tree: {}...", preview);
    }

    if parsed.len() > 5 {
        println!("   ... and {} more parsed files", parsed.len() - 5);
    }

    println!("\n🟫 Phase 3: Building Docpack Graph...");
    let mut graph = build_graph(&parsed);

    println!("\n🟧 Phase 4: Analyzing metrics (complexity, public API)...");
    let analysis_config = pipeline::analyze::AnalysisConfig::default();
    let analysis_result = pipeline::analyze::analyze_graph(&mut graph, &analysis_config)?;

    println!("   • Nodes analyzed: {}", analysis_result.nodes_analyzed);
    println!("   • Complexity calculated: {}", analysis_result.complexity_calculated);
    println!("   • Public API detected: {}", analysis_result.public_api_detected);

    let stats = graph.stats();
    println!("\n📊 Graph Statistics:");
    println!("   • Total nodes: {}", stats.total_nodes);
    println!("   • Total edges: {}", stats.total_edges);
    println!("   • Functions: {}", stats.functions);
    println!("   • Types: {}", stats.types);
    println!("   • Modules: {}", stats.modules);
    println!("   • Files: {}", stats.files);
    println!("   • Languages: {}", stats.languages);

    // Show some sample nodes
    println!("\n📦 Sample Nodes:");
    let mut count = 0;
    for node in graph.nodes.values().take(10) {
        println!(
            "   • {} ({}): {} @ {}:{}",
            node.name(),
            match &node.kind {
                graph::NodeKind::Function(_) => "function",
                graph::NodeKind::Type(_) => "type",
                graph::NodeKind::Module(_) => "module",
                graph::NodeKind::Constant(_) => "constant",
                graph::NodeKind::File(_) => "file",
                _ => "other",
            },
            if node.is_public() { "pub" } else { "priv" },
            node.location.file,
            node.location.start_line
        );
        count += 1;
    }
    if graph.nodes.len() > count {
        println!("   ... and {} more nodes", graph.nodes.len() - count);
    }

    // Show some sample edges
    println!("\n🔗 Sample Edges:");
    for edge in graph.edges.iter().take(10) {
        println!("   • {} {:?} {}", edge.source, edge.kind, edge.target);
    }
    if graph.edges.len() > 10 {
        println!("   ... and {} more edges", graph.edges.len() - 10);
    }

    println!("\n🟨 Phase 5: Semantic Clustering...");
    let cluster_config = pipeline::cluster::ClusterConfig::default();
    let cluster_result = pipeline::cluster::cluster_graph(&mut graph, &cluster_config)?;

    println!("\nClustering Results:");
    println!(
        "   • Embeddings generated: {}",
        cluster_result.embeddings_generated
    );
    println!("   • Clusters created: {}", cluster_result.clusters_created);
    println!(
        "   • Similarity edges: {}",
        cluster_result.similarity_edges_added
    );

    // Show sample clusters
    println!("\nSample Clusters:");
    let clusters: Vec<_> = graph
        .nodes
        .values()
        .filter(|n| matches!(n.kind, graph::NodeKind::Cluster(_)))
        .take(5)
        .collect();

    for cluster_node in clusters {
        if let graph::NodeKind::Cluster(c) = &cluster_node.kind {
            println!("   • {} ({} members)", c.name, c.members.len());
            if !c.keywords.is_empty() {
                println!("     Keywords: {}", c.keywords.join(", "));
            }
        }
    }

    // Ensure output directory exists, then save the graph to a file
    let output_dir = Path::new("output");
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
        println!("   ✓ Created output directory {:?}", output_dir);
    }

    let output_path = "output/docpack-graph.json";
    println!("\n💾 Saving graph to {}...", output_path);
    graph.save_to_file(output_path)?;
    println!("   ✓ Graph saved successfully!");

    println!("\n🟥 Phase 6: Generating documentation with LLM...");
    let gen_config = pipeline::generate::GenerationConfig::default();

    match pipeline::generate::generate_documentation(&graph, &gen_config).await {
        Ok(result) => {
            println!("\nDocumentation Generation Complete!");
            println!("   • Symbol summaries: {}", result.symbol_summaries.len());
            println!("   • Module overviews: {}", result.module_overviews.len());
            println!("   • Total tokens used: {}", result.total_tokens_used);

            // Show some sample generated docs
            println!("\nSample Generated Documentation:");
            for (node_id, doc) in result.symbol_summaries.iter().take(3) {
                println!("\n   Symbol: {}", node_id);
                println!("   Purpose: {}", doc.purpose);
                if !doc.explanation.is_empty() {
                    println!("   Explanation: {}", doc.explanation);
                }
            }

            // Show architecture overview
            println!("\nArchitecture Overview:");
            println!("   {}", result.architecture_overview.overview);

            // Save documentation to file
            let doc_path = "output/docpack-documentation.json";
            println!("\n💾 Saving documentation to {}...", doc_path);
            pipeline::generate::save_documentation(&result, doc_path)?;
            println!("   ✓ Documentation saved successfully!");

            println!("\nDocpack generation complete!");
            println!("   Graph: {}", output_path);
            println!("   Docs: {}", doc_path);
        }
        Err(e) => {
            eprintln!("\n❌ Error generating documentation: {}", e);
            eprintln!("   Make sure OPENAI_API_KEY is set in your .env file");
            eprintln!("   Continuing without LLM-generated documentation...");
        }
    }

    Ok(())
}
