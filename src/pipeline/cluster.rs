// pipeline/cluster.rs
// Phase 4: Semantic Clustering (Embedding Layer)
// This sits ON TOP of the AST graph

use crate::graph::{
    ClusterNode, ContentType, DocpackGraph, Edge, EdgeKind, EmbeddableContent, Embedding, Node,
    NodeId, NodeKind,
};
use hdbscan::{Hdbscan, HdbscanHyperParams};
use ndarray::Array2;

use ort::value::Value;
#[allow(unused)]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
    Error,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Configuration for clustering
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Minimum cluster size for HDBSCAN
    pub min_cluster_size: usize,

    /// Whether to use a mock embedding API (for testing without real API)
    pub use_mock_embeddings: bool,

    /// Path to ONNX model file
    pub model_path: String,

    /// Path to tokenizer JSON file
    pub tokenizer_path: String,

    /// Batch size for embedding generation
    pub batch_size: usize,

    /// Number of parallel worker threads (each with its own ONNX session)
    pub num_workers: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 2,
            use_mock_embeddings: false, // Use real embeddings by default
            model_path: "models/minilm-l6/model.onnx".to_string(),
            tokenizer_path: "models/minilm-l6/tokenizer.json".to_string(),
            batch_size: 8,
            num_workers: num_cpus::get().min(8).max(2), // Cap at 8 workers to avoid ORT issues
        }
    }
}

/// Perform semantic clustering on the graph
pub fn cluster_graph(
    graph: &mut DocpackGraph,
    config: &ClusterConfig,
) -> Result<ClusteringResult, Box<dyn std::error::Error>> {
    println!("Extracting embeddable content...");
    let contents = extract_embeddable_content(graph);
    println!("Found {} embeddable items", contents.len());

    if contents.is_empty() {
        return Ok(ClusteringResult {
            clusters_created: 0,
            embeddings_generated: 0,
            similarity_edges_added: 0,
        });
    }

    println!("   → Generating embeddings...");
    let embeddings = generate_embeddings(&contents, config)?;
    println!("      Generated {} embeddings", embeddings.len());

    println!("   → Running HDBSCAN clustering...");
    let cluster_assignments = run_clustering(&embeddings, config)?;

    println!("   → Creating cluster nodes...");
    let clusters_created =
        create_cluster_nodes(graph, &cluster_assignments, &contents, &embeddings)?;

    println!("   → Adding similarity edges...");
    let similarity_edges = add_similarity_edges(graph, &embeddings)?;

    Ok(ClusteringResult {
        clusters_created,
        embeddings_generated: embeddings.len(),
        similarity_edges_added: similarity_edges,
    })
}

/// Result of clustering operation
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct ClusteringResult {
    pub clusters_created: usize,
    pub embeddings_generated: usize,
    pub similarity_edges_added: usize,
}

/// Extract embeddable content from graph nodes
fn extract_embeddable_content(graph: &DocpackGraph) -> Vec<EmbeddableContent> {
    let mut contents = Vec::new();

    for node in graph.nodes.values() {
        match &node.kind {
            NodeKind::Function(f) => {
                // Extract function signature and docstring
                let mut text = format!("Function: {}\nSignature: {}", f.name, f.signature);

                if let Some(doc) = &node.metadata.docstring {
                    text.push_str(&format!("\nDocumentation: {}", doc));
                }

                contents.push(EmbeddableContent {
                    node_id: node.id.clone(),
                    text,
                    content_type: ContentType::FunctionBody,
                });

                // Also add docstring separately if it exists
                if let Some(doc) = &node.metadata.docstring {
                    contents.push(EmbeddableContent {
                        node_id: node.id.clone(),
                        text: doc.clone(),
                        content_type: ContentType::Docstring,
                    });
                }
            }

            NodeKind::Type(t) => {
                let mut text = format!("Type: {} ({:?})", t.name, t.kind);

                if !t.fields.is_empty() {
                    text.push_str("\nFields: ");
                    for field in &t.fields {
                        text.push_str(&format!("{}, ", field.name));
                    }
                }

                if let Some(doc) = &node.metadata.docstring {
                    text.push_str(&format!("\nDocumentation: {}", doc));
                }

                contents.push(EmbeddableContent {
                    node_id: node.id.clone(),
                    text,
                    content_type: ContentType::TypeDefinition,
                });
            }

            NodeKind::Module(m) => {
                let mut text = format!("Module: {}\nPath: {}", m.name, m.path);

                if let Some(doc) = &node.metadata.docstring {
                    text.push_str(&format!("\nDocumentation: {}", doc));
                }

                contents.push(EmbeddableContent {
                    node_id: node.id.clone(),
                    text,
                    content_type: ContentType::ModuleContent,
                });
            }

            _ => {}
        }
    }

    contents
}

/// Generate embeddings for content using Jina embeddings with batching
fn generate_embeddings(
    contents: &[EmbeddableContent],
    config: &ClusterConfig,
) -> Result<Vec<Embedding>, Box<dyn std::error::Error>> {
    if config.use_mock_embeddings {
        // Generate deterministic mock embeddings based on content hash
        Ok(contents
            .par_iter()
            .map(|content| generate_mock_embedding(content))
            .collect())
    } else {
        // Use Jina embeddings with batching for performance
        generate_jina_embeddings(contents, config)
    }
}

/// Generate embeddings using ONNX model with dedicated worker threads
fn generate_jina_embeddings(
    contents: &[EmbeddableContent],
    config: &ClusterConfig,
) -> Result<Vec<Embedding>, Box<dyn std::error::Error>> {
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::Instant;

    println!("      Loading {} ONNX workers...", config.num_workers);
    let load_start = Instant::now();

    // Load tokenizer once (shared Arc, read-only so safe)
    let tokenizer = Arc::new(
        Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
    );

    // Create work queue and result channels
    let (work_tx, work_rx) =
        mpsc::sync_channel::<Option<(usize, Vec<EmbeddableContent>)>>(config.num_workers * 2);
    let work_rx = Arc::new(std::sync::Mutex::new(work_rx));
    let (result_tx, result_rx) = mpsc::channel::<(usize, Vec<Embedding>)>();

    // Spawn worker threads, each with its own ONNX session
    let mut workers = Vec::new();
    for worker_id in 0..config.num_workers {
        let tokenizer = Arc::clone(&tokenizer);
        let work_rx = Arc::clone(&work_rx);
        let result_tx = result_tx.clone();
        let model_path = config.model_path.clone();

        let worker = thread::spawn(move || -> Result<(), String> {
            // Each worker loads its own ONNX session
            let mut session = Session::builder()
                .map_err(|e| format!("Worker {} session builder error: {:?}", worker_id, e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| format!("Worker {} optimization error: {:?}", worker_id, e))?
                .with_intra_threads(2)
                .map_err(|e| format!("Worker {} thread config error: {:?}", worker_id, e))?
                .commit_from_file(&model_path)
                .map_err(|e| format!("Worker {} model load error: {:?}", worker_id, e))?;

            // Process work items from queue
            loop {
                let work_item = {
                    let rx = work_rx.lock().unwrap();
                    rx.recv()
                        .map_err(|e| format!("Worker {} channel error: {:?}", worker_id, e))?
                };

                match work_item {
                    None => break, // Shutdown signal
                    Some((batch_idx, chunk)) => {
                        let batch_start = Instant::now();

                        // Tokenize batch
                        let texts: Vec<&str> = chunk.iter().map(|c| c.text.as_str()).collect();
                        let encodings =
                            tokenizer.encode_batch(texts.clone(), true).map_err(|e| {
                                format!("Worker {} tokenization failed: {:?}", worker_id, e)
                            })?;

                        // Prepare input tensors
                        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
                        let batch_size = encodings.len();

                        let mut input_ids = vec![0i64; batch_size * max_len];
                        let mut attention_mask = vec![0i64; batch_size * max_len];
                        let mut token_type_ids = vec![0i64; batch_size * max_len];

                        for (i, encoding) in encodings.iter().enumerate() {
                            let ids = encoding.get_ids();
                            let mask = encoding.get_attention_mask();
                            let type_ids = encoding.get_type_ids();
                            for (j, &id) in ids.iter().enumerate() {
                                input_ids[i * max_len + j] = id as i64;
                                attention_mask[i * max_len + j] = mask[j] as i64;
                                token_type_ids[i * max_len + j] = type_ids[j] as i64;
                            }
                        }

                        // Create ndarray inputs
                        let input_ids_array =
                            Array2::from_shape_vec((batch_size, max_len), input_ids).map_err(
                                |e| format!("Worker {} shape error: {:?}", worker_id, e),
                            )?;
                        let attention_mask_array =
                            Array2::from_shape_vec((batch_size, max_len), attention_mask).map_err(
                                |e| format!("Worker {} shape error: {:?}", worker_id, e),
                            )?;
                        let token_type_ids_array =
                            Array2::from_shape_vec((batch_size, max_len), token_type_ids).map_err(
                                |e| format!("Worker {} shape error: {:?}", worker_id, e),
                            )?;

                        // Run inference
                        let input_ids_value = Value::from_array(input_ids_array).map_err(|e| {
                            format!("Worker {} value creation error: {:?}", worker_id, e)
                        })?;
                        let attention_mask_value = Value::from_array(attention_mask_array)
                            .map_err(|e| {
                                format!("Worker {} value creation error: {:?}", worker_id, e)
                            })?;
                        let token_type_ids_value = Value::from_array(token_type_ids_array)
                            .map_err(|e| {
                                format!("Worker {} value creation error: {:?}", worker_id, e)
                            })?;

                        let outputs = session
                            .run(ort::inputs![
                                "input_ids" => &input_ids_value,
                                "attention_mask" => &attention_mask_value,
                                "token_type_ids" => &token_type_ids_value
                            ])
                            .map_err(|e| {
                                format!("Worker {} ONNX inference error: {:?}", worker_id, e)
                            })?;

                        // Extract embeddings (mean pooling)
                        let (output_shape, output_data) =
                            outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                                format!("Worker {} tensor extraction error: {:?}", worker_id, e)
                            })?;
                        let embedding_dim = output_shape[2];

                        // Get mask arrays for mean pooling
                        let (_, attention_mask_data) = attention_mask_value
                            .try_extract_tensor::<i64>()
                            .map_err(|e| {
                                format!("Worker {} mask extraction error: {:?}", worker_id, e)
                            })?;

                        let mut batch_embeddings = Vec::new();

                        for (i, content) in chunk.iter().enumerate() {
                            // Mean pooling over sequence length
                            let mut pooled = vec![0.0f32; embedding_dim as usize];
                            let mut count = 0;

                            for j in 0..max_len {
                                let mask_idx = i * max_len + j;
                                if attention_mask_data[mask_idx] == 1 {
                                    for k in 0..embedding_dim {
                                        let tensor_idx = i * max_len * (embedding_dim as usize)
                                            + j * (embedding_dim as usize)
                                            + (k as usize);
                                        pooled[k as usize] += output_data[tensor_idx as usize];
                                    }
                                    count += 1;
                                }
                            }

                            if count > 0 {
                                for val in &mut pooled {
                                    *val /= count as f32;
                                }
                            }

                            // Normalize
                            let magnitude: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if magnitude > 0.0 {
                                for val in &mut pooled {
                                    *val /= magnitude;
                                }
                            }

                            batch_embeddings.push(Embedding {
                                node_id: content.node_id.clone(),
                                vector: pooled,
                                model: "minilm-l6-v2".to_string(),
                            });
                        }

                        println!(
                            "           ✓ Worker {} batch {}: {} items in {:.2}s ({:.0} items/sec)",
                            worker_id,
                            batch_idx + 1,
                            texts.len(),
                            batch_start.elapsed().as_secs_f32(),
                            texts.len() as f32 / batch_start.elapsed().as_secs_f32()
                        );

                        // Send results back
                        result_tx.send((batch_idx, batch_embeddings)).map_err(|e| {
                            format!("Worker {} result send error: {:?}", worker_id, e)
                        })?;
                    }
                }
            }

            Ok(())
        });

        workers.push(worker);
    }

    println!(
        "      ✓ Loaded {} workers in {:.2}s",
        config.num_workers,
        load_start.elapsed().as_secs_f32()
    );

    // Send work items to queue
    let total_batches = (contents.len() + config.batch_size - 1) / config.batch_size;
    println!(
        "      Processing {} items in {} batches of {} across {} workers",
        contents.len(),
        total_batches,
        config.batch_size,
        config.num_workers
    );

    let total_start = Instant::now();

    // Dispatch all work
    for (batch_idx, chunk) in contents.chunks(config.batch_size).enumerate() {
        work_tx
            .send(Some((batch_idx, chunk.to_vec())))
            .map_err(|e| format!("Failed to send work: {:?}", e))?;
    }

    // Send shutdown signals
    for _ in 0..config.num_workers {
        work_tx.send(None).ok();
    }
    drop(work_tx); // Close the channel

    // Collect results
    let mut results = vec![Vec::new(); total_batches];
    for _ in 0..total_batches {
        let (batch_idx, embeddings) = result_rx
            .recv()
            .map_err(|e| format!("Failed to receive result: {:?}", e))?;
        results[batch_idx] = embeddings;
    }

    // Wait for all workers to finish
    for worker in workers {
        worker
            .join()
            .map_err(|e| format!("Worker thread panicked: {:?}", e))?
            .map_err(|e| format!("Worker error: {}", e))?;
    }

    // Flatten results
    let all_embeddings: Vec<Embedding> = results.into_iter().flatten().collect();

    println!(
        "      ✓ Generated {} embeddings in {:.2}s ({:.0} items/sec overall)",
        all_embeddings.len(),
        total_start.elapsed().as_secs_f32(),
        all_embeddings.len() as f32 / total_start.elapsed().as_secs_f32()
    );

    Ok(all_embeddings)
}

/// Generate a deterministic mock embedding from content
fn generate_mock_embedding(content: &EmbeddableContent) -> Embedding {
    // Create a deterministic embedding based on content hash
    // This is just for testing - real embeddings would come from a model

    let mut hasher = Sha256::new();
    hasher.update(content.text.as_bytes());
    let hash = hasher.finalize();

    // Convert hash to 384-dimensional vector (typical embedding size)
    let mut vector = Vec::with_capacity(384);
    for chunk in hash.chunks(4) {
        let val = u32::from_le_bytes([
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
            chunk.get(3).copied().unwrap_or(0),
        ]);
        vector.push((val % 1000) as f32 / 1000.0);
    }

    // Pad to 384 dimensions
    while vector.len() < 384 {
        vector.push(0.1);
    }

    // Add some variation based on content type
    let type_offset = match content.content_type {
        ContentType::FunctionBody => 0.1,
        ContentType::Docstring => 0.2,
        ContentType::TypeDefinition => 0.3,
        ContentType::ModuleContent => 0.4,
        ContentType::Comment => 0.5,
    };

    for i in 0..10 {
        vector[i] += type_offset;
    }

    // Normalize vector
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for v in &mut vector {
            *v /= magnitude;
        }
    }

    Embedding {
        node_id: content.node_id.clone(),
        vector,
        model: "mock-embeddings-v1".to_string(),
    }
}

/// Run HDBSCAN clustering on embeddings
fn run_clustering(
    embeddings: &[Embedding],
    config: &ClusterConfig,
) -> Result<Vec<(NodeId, i32)>, Box<dyn std::error::Error>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    // Convert embeddings to Vec<Vec<f32>> format for HDBSCAN
    let data: Vec<Vec<f32>> = embeddings.iter().map(|emb| emb.vector.clone()).collect();

    // Run HDBSCAN with hyperparameters
    let params = HdbscanHyperParams::builder()
        .min_cluster_size(config.min_cluster_size)
        .build();

    let clusterer = Hdbscan::new(&data, params);
    let labels = clusterer.cluster()?;

    // Map labels back to node IDs
    let assignments: Vec<(NodeId, i32)> = embeddings
        .iter()
        .zip(labels.iter())
        .map(|(emb, &label)| (emb.node_id.clone(), label))
        .collect();

    Ok(assignments)
}

/// Create cluster nodes in the graph
fn create_cluster_nodes(
    graph: &mut DocpackGraph,
    assignments: &[(NodeId, i32)],
    contents: &[EmbeddableContent],
    embeddings: &[Embedding],
) -> Result<usize, Box<dyn std::error::Error>> {
    // Group nodes by cluster ID
    let mut clusters: HashMap<i32, Vec<NodeId>> = HashMap::new();

    for (node_id, cluster_id) in assignments {
        if *cluster_id >= 0 {
            // -1 indicates noise/outliers in HDBSCAN
            clusters
                .entry(*cluster_id)
                .or_insert_with(Vec::new)
                .push(node_id.clone());
        }
    }

    let mut created_count = 0;

    for (cluster_id, members) in clusters {
        if members.len() < 2 {
            continue; // Skip singleton clusters
        }

        // Generate cluster name and keywords
        let (cluster_name, keywords) = generate_cluster_metadata(&members, contents, graph);

        // Compute centroid
        let centroid = compute_centroid(&members, embeddings);

        // Create cluster node
        let cluster_node_id = format!("cluster_{}", cluster_id);
        let cluster_node = Node {
            id: cluster_node_id.clone(),
            kind: NodeKind::Cluster(ClusterNode {
                name: cluster_name,
                topic: None, // Could be enhanced with topic modeling
                members: members.clone(),
                keywords,
                centroid: Some(centroid),
            }),
            location: crate::graph::Location {
                file: "virtual".to_string(),
                start_line: 0,
                end_line: 0,
                start_col: 0,
                end_col: 0,
            },
            metadata: Default::default(),
        };

        graph.add_node(cluster_node);

        // Add edges from cluster to members
        for member_id in &members {
            graph.add_edge(Edge::new(
                cluster_node_id.clone(),
                member_id.clone(),
                EdgeKind::ModuleOwnership, // Reusing this edge type
            ));
        }

        created_count += 1;
    }

    Ok(created_count)
}

/// Generate cluster name and keywords from member nodes
fn generate_cluster_metadata(
    members: &[NodeId],
    contents: &[EmbeddableContent],
    graph: &DocpackGraph,
) -> (String, Vec<String>) {
    // Extract common keywords from member names and content
    let mut keywords = Vec::new();
    let mut name_parts = Vec::new();

    for member_id in members.iter().take(5) {
        // Look at first 5 members
        if let Some(node) = graph.nodes.get(member_id) {
            let name = node.name();
            name_parts.push(name.clone());

            // Extract words from name (split on underscores, camelCase, etc.)
            let words = extract_keywords(&name);
            keywords.extend(words);
        }
    }

    // Also extract from content
    for content in contents.iter().filter(|c| members.contains(&c.node_id)) {
        let words = extract_keywords(&content.text);
        keywords.extend(words.into_iter().take(3)); // Limit per content
    }

    // Count keyword frequency
    let mut keyword_counts: HashMap<String, usize> = HashMap::new();
    for keyword in keywords {
        *keyword_counts.entry(keyword).or_insert(0) += 1;
    }

    // Get top keywords
    let mut sorted_keywords: Vec<_> = keyword_counts.into_iter().collect();
    sorted_keywords.sort_by(|a, b| b.1.cmp(&a.1));
    let top_keywords: Vec<String> = sorted_keywords
        .into_iter()
        .take(5)
        .map(|(k, _)| k)
        .collect();

    // Generate cluster name from top keywords
    let cluster_name = if top_keywords.is_empty() {
        format!("Cluster with {} members", members.len())
    } else {
        format!("{} module", top_keywords[0])
    };

    (cluster_name, top_keywords)
}

/// Extract keywords from text
fn extract_keywords(text: &str) -> Vec<String> {
    let mut keywords = Vec::new();

    // Split on common delimiters
    let words = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2) // Skip very short words
        .map(|w| w.to_lowercase());

    for word in words {
        // Skip common words
        if is_stopword(&word) {
            continue;
        }
        keywords.push(word);
    }

    keywords
}

/// Check if word is a stopword
fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "and"
            | "for"
            | "that"
            | "this"
            | "with"
            | "from"
            | "have"
            | "has"
            | "are"
            | "was"
            | "were"
            | "been"
    )
}

/// Compute centroid of embeddings
fn compute_centroid(members: &[NodeId], embeddings: &[Embedding]) -> Vec<f32> {
    if members.is_empty() {
        return vec![];
    }

    // Get embeddings for members
    let member_embeddings: Vec<&Embedding> = embeddings
        .iter()
        .filter(|e| members.contains(&e.node_id))
        .collect();

    if member_embeddings.is_empty() {
        return vec![];
    }

    let dim = member_embeddings[0].vector.len();
    let mut centroid = vec![0.0; dim];

    // Sum all vectors
    for emb in &member_embeddings {
        for (i, &val) in emb.vector.iter().enumerate() {
            centroid[i] += val;
        }
    }

    // Average
    let count = member_embeddings.len() as f32;
    for val in &mut centroid {
        *val /= count;
    }

    centroid
}

/// Add similarity edges between related nodes
fn add_similarity_edges(
    graph: &mut DocpackGraph,
    embeddings: &[Embedding],
) -> Result<usize, Box<dyn std::error::Error>> {
    let threshold = 0.75; // Cosine similarity threshold
    let mut edges_added = 0;

    // Compare each pair of embeddings
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = cosine_similarity(&embeddings[i].vector, &embeddings[j].vector);

            if sim > threshold {
                // Add similarity edge (using DataFlow as generic relationship)
                graph.add_edge(Edge::new(
                    embeddings[i].node_id.clone(),
                    embeddings[j].node_id.clone(),
                    EdgeKind::DataFlow,
                ));
                edges_added += 1;
            }
        }
    }

    Ok(edges_added)
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}
