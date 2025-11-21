// Pipeline Phase 5: LLM Documentation Generation
// Uses OpenAI to generate human-readable documentation from graph facts

use crate::graph::{DocpackGraph, EdgeKind, Node, NodeKind};
use futures::future::join_all;
use openai::Credentials;
use openai::chat::{
    ChatCompletion, ChatCompletionMessage, ChatCompletionMessageRole, ChatCompletionResponseFormat,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for LLM generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub credentials: Credentials,
    pub model: String,
    pub max_completion_tokens: u32,
    pub temperature: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        // Note: The openai crate expects OPENAI_KEY not OPENAI_API_KEY
        let api_key = std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("OPENAI_KEY"))
            .unwrap_or_default();
        let base_url = std::env::var("OPENAI_BASE_URL").unwrap_or_default();

        Self {
            credentials: Credentials::new(api_key, base_url),
            model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            max_completion_tokens: 8000, // 8k token limit per batch
            temperature: 1.0,
        }
    }
}

/// Result of documentation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub symbol_summaries: HashMap<String, SymbolDoc>,
    pub module_overviews: HashMap<String, ModuleDoc>,
    pub architecture_overview: ArchitectureDoc,
    pub total_tokens_used: usize,
}

/// Documentation for a single symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolDoc {
    pub node_id: String,
    pub purpose: String,
    pub explanation: String,
    pub complexity_notes: Option<String>,
    pub usage_hints: Option<String>,
}

/// Documentation for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDoc {
    pub module_name: String,
    pub responsibilities: String,
    pub key_symbols: Vec<String>,
    pub interactions: String,
}

/// Architecture-level documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureDoc {
    pub overview: String,
    pub system_behavior: String,
    pub data_flow: String,
    pub key_components: Vec<String>,
}

/// Generate all documentation for a graph
pub async fn generate_documentation(
    graph: &DocpackGraph,
    config: &GenerationConfig,
) -> Result<GenerationResult, Box<dyn std::error::Error>> {
    println!(
        "   🤖 Generating documentation with OpenAI {}...",
        config.model
    );

    let mut total_tokens = 0;

    // Step 1: Generate symbol summaries
    println!("   📝 Generating symbol summaries...");
    let (symbol_summaries, tokens) = generate_symbol_summaries(graph, config).await?;
    total_tokens += tokens;
    println!(
        "      ✓ Generated {} symbol summaries ({} tokens)",
        symbol_summaries.len(),
        tokens
    );

    // Step 2: Generate module overviews
    println!("   📦 Generating module overviews...");
    let (module_overviews, tokens) = generate_module_overviews(graph, config).await?;
    total_tokens += tokens;
    println!(
        "      ✓ Generated {} module overviews ({} tokens)",
        module_overviews.len(),
        tokens
    );

    // Step 3: Generate architecture overview
    println!("   🏗️  Generating architecture overview...");
    let (architecture_overview, tokens) = generate_architecture_overview(graph, config).await?;
    total_tokens += tokens;
    println!(
        "      ✓ Generated architecture overview ({} tokens)",
        tokens
    );

    Ok(GenerationResult {
        symbol_summaries,
        module_overviews,
        architecture_overview,
        total_tokens_used: total_tokens,
    })
}

/// Generate summaries for important symbols (functions, types)
async fn generate_symbol_summaries(
    graph: &DocpackGraph,
    config: &GenerationConfig,
) -> Result<(HashMap<String, SymbolDoc>, usize), Box<dyn std::error::Error>> {
    let mut summaries = HashMap::new();
    let mut total_tokens = 0;

    // Get functions and types, prioritizing public API
    let mut important_nodes: Vec<&Node> = graph
        .nodes
        .values()
        .filter(|n| matches!(n.kind, NodeKind::Function(_)) || matches!(n.kind, NodeKind::Type(_)))
        .collect();

    // Sort by public API first, then by complexity
    important_nodes.sort_by(
        |a, b| match (a.metadata.is_public_api, b.metadata.is_public_api) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b
                .metadata
                .complexity
                .unwrap_or(0)
                .cmp(&a.metadata.complexity.unwrap_or(0)),
        },
    );

    // Limit to top 50 symbols to control costs
    important_nodes.truncate(50);

    // Create dynamic batches based on token estimates (target ~6k tokens per batch)
    let batches = create_dynamic_batches(graph, &important_nodes, 6000);
    println!(
        "      Created {} batches for parallel processing",
        batches.len()
    );

    // Process batches in parallel
    let batch_futures: Vec<_> = batches
        .iter()
        .map(|batch| generate_symbol_batch(graph, batch.as_slice(), config))
        .collect();

    let results = join_all(batch_futures).await;

    // Collect results
    for result in results {
        let (batch_summaries, tokens) = result?;
        total_tokens += tokens;
        summaries.extend(batch_summaries);
    }

    Ok((summaries, total_tokens))
}

/// Generate summaries for a batch of symbols
async fn generate_symbol_batch(
    graph: &DocpackGraph,
    nodes: &[&Node],
    config: &GenerationConfig,
) -> Result<(HashMap<String, SymbolDoc>, usize), Box<dyn std::error::Error>> {
    if nodes.is_empty() {
        return Ok((HashMap::new(), 0));
    }

    let mut summaries = HashMap::new();

    // Build prompt with facts about each symbol
    let mut prompt = String::from(
        "You are a technical documentation writer. For each symbol below, provide a concise, accurate description based ONLY on the facts provided. Do not infer or assume anything.\n\n",
    );

    for (idx, node) in nodes.iter().enumerate() {
        prompt.push_str(&format!("\n=== SYMBOL {} ===\n", idx + 1));
        prompt.push_str(&format_symbol_facts(graph, node));
    }

    prompt.push_str("\n\nProvide a JSON object with a 'symbols' array. Each entry should have:\n");
    prompt.push_str("- symbol_index: The symbol number (1-based)\n");
    prompt.push_str("- purpose: A one-sentence purpose statement\n");
    prompt.push_str("- explanation: A brief explanation (2-3 sentences) of what it does\n");
    prompt.push_str("- complexity_notes: Any complexity notes if complexity > 10 (or null)\n");
    prompt.push_str("- usage_hints: Usage hints based on call patterns (or null)\n\n");
    prompt.push_str("Return valid JSON matching this structure:\n");
    prompt.push_str(r#"{"symbols": [{"symbol_index": 1, "purpose": "...", "explanation": "...", "complexity_notes": null, "usage_hints": "..."}]}"#);

    // Call OpenAI API with JSON mode
    let response = call_openai_json(config, &prompt).await?;
    let tokens = estimate_tokens(&prompt) + estimate_tokens(&response);

    // Parse JSON response
    let parsed = parse_symbol_json_response(&response, nodes)?;
    summaries.extend(parsed);

    Ok((summaries, tokens))
}

/// Format facts about a symbol for the LLM
fn format_symbol_facts(graph: &DocpackGraph, node: &Node) -> String {
    let mut facts = String::new();

    facts.push_str(&format!("ID: {}\n", node.id));
    facts.push_str(&format!("Name: {}\n", node.name()));
    facts.push_str(&format!(
        "Location: {}:{}\n",
        node.location.file, node.location.start_line
    ));
    facts.push_str(&format!("Public API: {}\n", node.metadata.is_public_api));

    match &node.kind {
        NodeKind::Function(f) => {
            facts.push_str(&format!("Type: Function\n"));
            facts.push_str(&format!("Signature: {}\n", f.signature));
            facts.push_str(&format!("Parameters: {}\n", f.parameters.len()));
            if let Some(ret) = &f.return_type {
                facts.push_str(&format!("Returns: {}\n", ret));
            }
            facts.push_str(&format!("Async: {}\n", f.is_async));
        }
        NodeKind::Type(t) => {
            facts.push_str(&format!("Type: {:?}\n", t.kind));
            facts.push_str(&format!("Fields: {}\n", t.fields.len()));
            facts.push_str(&format!("Methods: {}\n", t.methods.len()));
        }
        _ => {}
    }

    // Add complexity
    if let Some(complexity) = node.metadata.complexity {
        facts.push_str(&format!("Complexity: {}\n", complexity));
    }

    // Add fan-in/fan-out
    facts.push_str(&format!(
        "Fan-in: {} (things that call/use this)\n",
        node.metadata.fan_in
    ));
    facts.push_str(&format!(
        "Fan-out: {} (things this calls/uses)\n",
        node.metadata.fan_out
    ));

    // Add docstring if available
    if let Some(doc) = &node.metadata.docstring {
        facts.push_str(&format!("Existing Doc: {}\n", doc));
    }

    // Add call relationships
    let callers: Vec<_> = graph
        .get_incoming_edges(&node.id)
        .iter()
        .filter(|e| matches!(e.kind, EdgeKind::Calls))
        .map(|e| &e.source)
        .take(3)
        .collect();

    if !callers.is_empty() {
        facts.push_str(&format!(
            "Called by: {}\n",
            callers
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let callees: Vec<_> = graph
        .get_outgoing_edges(&node.id)
        .iter()
        .filter(|e| matches!(e.kind, EdgeKind::Calls))
        .map(|e| &e.target)
        .take(3)
        .collect();

    if !callees.is_empty() {
        facts.push_str(&format!(
            "Calls: {}\n",
            callees
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    facts
}

/// JSON structure for symbol batch response
#[derive(Debug, Deserialize)]
struct SymbolBatchResponse {
    symbols: Vec<SymbolResponseItem>,
}

#[derive(Debug, Deserialize)]
struct SymbolResponseItem {
    symbol_index: usize,
    purpose: String,
    explanation: String,
    complexity_notes: Option<String>,
    usage_hints: Option<String>,
}

/// Parse JSON response for symbol summaries
fn parse_symbol_json_response(
    response: &str,
    nodes: &[&Node],
) -> Result<HashMap<String, SymbolDoc>, Box<dyn std::error::Error>> {
    let mut summaries = HashMap::new();

    let batch: SymbolBatchResponse = serde_json::from_str(response).map_err(|e| {
        eprintln!("Failed to parse JSON response. Error: {}", e);
        eprintln!(
            "Response (first 500 chars): {}",
            &response.chars().take(500).collect::<String>()
        );
        format!("JSON parsing error: {}", e)
    })?;

    for item in batch.symbols {
        let idx = item.symbol_index;
        if idx > 0 && idx <= nodes.len() {
            let node_id = nodes[idx - 1].id.clone();
            summaries.insert(
                node_id.clone(),
                SymbolDoc {
                    node_id,
                    purpose: item.purpose,
                    explanation: item.explanation,
                    complexity_notes: item.complexity_notes,
                    usage_hints: item.usage_hints,
                },
            );
        }
    }

    Ok(summaries)
}

/// Generate module overviews
async fn generate_module_overviews(
    graph: &DocpackGraph,
    config: &GenerationConfig,
) -> Result<(HashMap<String, ModuleDoc>, usize), Box<dyn std::error::Error>> {
    let mut overviews = HashMap::new();
    let mut total_tokens = 0;

    // Get all modules
    let modules: Vec<&Node> = graph
        .nodes
        .values()
        .filter(|n| matches!(n.kind, NodeKind::Module(_)))
        .take(20) // Limit modules
        .collect();

    // Process modules in parallel
    let module_futures: Vec<_> = modules
        .iter()
        .map(|module| generate_module_doc(graph, module, config))
        .collect();

    let results = join_all(module_futures).await;

    // Collect results
    for (module, result) in modules.iter().zip(results) {
        let (doc, tokens) = result?;
        total_tokens += tokens;
        overviews.insert(module.id.clone(), doc);
    }

    Ok((overviews, total_tokens))
}

/// Generate documentation for a single module
async fn generate_module_doc(
    graph: &DocpackGraph,
    module: &Node,
    config: &GenerationConfig,
) -> Result<(ModuleDoc, usize), Box<dyn std::error::Error>> {
    let module_data = if let NodeKind::Module(m) = &module.kind {
        m
    } else {
        return Err("Not a module node".into());
    };

    // Find symbols that belong to this module by examining the module's file location
    // Since module.children is not populated, we need to find symbols in the same file
    let module_file = &module.location.file;
    let module_symbols: Vec<&Node> = graph
        .nodes
        .values()
        .filter(|n| {
            // Include symbols from the same file that aren't modules themselves
            n.location.file == *module_file && !matches!(n.kind, NodeKind::Module(_))
        })
        .collect();

    // Build facts about the module
    let mut prompt = format!(
        "You are a technical documentation writer. Describe this module based ONLY on the provided facts.\n\n"
    );
    prompt.push_str(&format!("MODULE: {}\n", module_data.name));
    prompt.push_str(&format!("Path: {}\n", module_data.path));
    prompt.push_str(&format!("Public: {}\n", module_data.is_public));
    prompt.push_str(&format!("Child symbols: {}\n", module_symbols.len()));

    // List key symbols
    let key_symbols: Vec<String> = module_symbols
        .iter()
        .filter(|n| n.metadata.is_public_api || n.is_public())
        .take(10)
        .map(|n| {
            format!(
                "{} ({})",
                n.name(),
                match n.kind {
                    NodeKind::Function(_) => "fn",
                    NodeKind::Type(_) => "type",
                    _ => "other",
                }
            )
        })
        .collect();

    if !key_symbols.is_empty() {
        prompt.push_str(&format!("\nKey public symbols:\n"));
        for sym in &key_symbols {
            prompt.push_str(&format!("  - {}\n", sym));
        }
    }

    // Module dependencies
    let imports: Vec<_> = graph
        .get_outgoing_edges(&module.id)
        .iter()
        .filter(|e| matches!(e.kind, EdgeKind::Imports))
        .map(|e| &e.target)
        .take(5)
        .collect();

    if !imports.is_empty() {
        prompt.push_str(&format!(
            "\nImports: {}\n",
            imports
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    prompt.push_str("\n\nProvide:\n");
    prompt.push_str("1. Module responsibilities (1-2 sentences)\n");
    prompt.push_str("2. How it interacts with other modules\n");
    prompt.push_str("\nFormat:\nResponsibilities: ...\nInteractions: ...\n");

    let response = call_openai(config, &prompt).await?;
    let tokens = estimate_tokens(&prompt) + estimate_tokens(&response);

    // Parse response
    let responsibilities = extract_field(&response, "Responsibilities:");
    let interactions = extract_field(&response, "Interactions:");

    Ok((
        ModuleDoc {
            module_name: module_data.name.clone(),
            responsibilities,
            key_symbols: key_symbols.iter().take(5).cloned().collect(),
            interactions,
        },
        tokens,
    ))
}

/// Generate architecture overview
async fn generate_architecture_overview(
    graph: &DocpackGraph,
    config: &GenerationConfig,
) -> Result<(ArchitectureDoc, usize), Box<dyn std::error::Error>> {
    let stats = graph.stats();

    let mut prompt = String::from(
        "You are a senior software architect. Provide a high-level architecture overview based ONLY on these facts:\n\n",
    );

    prompt.push_str(&format!("REPOSITORY STATISTICS:\n"));
    prompt.push_str(&format!("- Total symbols: {}\n", stats.total_nodes));
    prompt.push_str(&format!("- Functions: {}\n", stats.functions));
    prompt.push_str(&format!("- Types: {}\n", stats.types));
    prompt.push_str(&format!("- Modules: {}\n", stats.modules));
    prompt.push_str(&format!("- Files: {}\n", stats.files));
    prompt.push_str(&format!("- Languages: {}\n", stats.languages));
    prompt.push_str(&format!("- Relationships: {}\n", stats.total_edges));

    // List major modules
    let modules: Vec<String> = graph
        .nodes
        .values()
        .filter(|n| matches!(n.kind, NodeKind::Module(_)))
        .take(10)
        .map(|n| n.name())
        .collect();

    if !modules.is_empty() {
        prompt.push_str(&format!("\nMAJOR MODULES:\n"));
        for module in modules {
            prompt.push_str(&format!("  - {}\n", module));
        }
    }

    // List clusters if available
    let clusters: Vec<String> = graph
        .nodes
        .values()
        .filter(|n| matches!(n.kind, NodeKind::Cluster(_)))
        .take(10)
        .map(|n| n.name())
        .collect();

    if !clusters.is_empty() {
        prompt.push_str(&format!("\nSEMANTIC CLUSTERS:\n"));
        for cluster in clusters {
            prompt.push_str(&format!("  - {}\n", cluster));
        }
    }

    prompt.push_str("\n\nProvide:\n");
    prompt.push_str("1. High-level architectural overview (2-3 sentences)\n");
    prompt.push_str("2. System behavior and purpose (2-3 sentences)\n");
    prompt.push_str("3. Data flow patterns (1-2 sentences)\n");
    prompt.push_str("4. Key components (list 3-5)\n");
    prompt.push_str(
        "\nFormat:\nOverview: ...\nBehavior: ...\nData Flow: ...\nKey Components:\n- ...\n- ...\n",
    );

    let response = call_openai(config, &prompt).await?;
    let tokens = estimate_tokens(&prompt) + estimate_tokens(&response);

    // Parse response
    let overview = extract_field(&response, "Overview:");
    let behavior = extract_field(&response, "Behavior:");
    let data_flow = extract_field(&response, "Data Flow:");
    let key_components = extract_list(&response, "Key Components:");

    Ok((
        ArchitectureDoc {
            overview,
            system_behavior: behavior,
            data_flow,
            key_components,
        },
        tokens,
    ))
}

/// Call OpenAI API
async fn call_openai(
    config: &GenerationConfig,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![
        ChatCompletionMessage {
            role: ChatCompletionMessageRole::System,
            content: Some("You are a precise technical documentation writer. Base your responses ONLY on the provided facts. Never infer or hallucinate information.".to_string()),
            name: None,
            function_call: None,
            tool_call_id: None,
            tool_calls: None,
        },
        ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(prompt.to_string()),
            name: None,
            function_call: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let response = ChatCompletion::builder(&config.model, messages)
        .credentials(config.credentials.clone())
        .max_completion_tokens(config.max_completion_tokens)
        .temperature(config.temperature)
        .create()
        .await?;

    let content = response
        .choices
        .get(0)
        .and_then(|choice| choice.message.content.as_ref())
        .ok_or("No response content")?;

    Ok(content.clone())
}

/// Call OpenAI API with JSON mode
async fn call_openai_json(
    config: &GenerationConfig,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let messages = vec![
        ChatCompletionMessage {
            role: ChatCompletionMessageRole::System,
            content: Some("You are a precise technical documentation writer. Base your responses ONLY on the provided facts. Never infer or hallucinate information. Always respond with valid JSON.".to_string()),
            name: None,
            function_call: None,
            tool_call_id: None,
            tool_calls: None,
        },
        ChatCompletionMessage {
            role: ChatCompletionMessageRole::User,
            content: Some(prompt.to_string()),
            name: None,
            function_call: None,
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let response = ChatCompletion::builder(&config.model, messages)
        .credentials(config.credentials.clone())
        .max_completion_tokens(config.max_completion_tokens)
        .temperature(config.temperature)
        .response_format(ChatCompletionResponseFormat::json_object())
        .create()
        .await
        .map_err(|e| format!("OpenAI API error: {}", e))?;

    let content = response
        .choices
        .get(0)
        .and_then(|choice| choice.message.content.as_ref())
        .ok_or("No response content from OpenAI")?;

    Ok(content.clone())
}

/// Estimate token count (rough approximation)
fn estimate_tokens(text: &str) -> usize {
    // Rough estimate: ~4 chars per token
    text.len() / 4
}

/// Create dynamic batches of nodes based on token estimates
fn create_dynamic_batches<'a>(
    graph: &DocpackGraph,
    nodes: &'a [&'a Node],
    target_tokens: usize,
) -> Vec<Vec<&'a Node>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_tokens = 0;

    for node in nodes {
        let node_facts = format_symbol_facts(graph, node);
        let node_tokens = estimate_tokens(&node_facts);

        // If adding this node would exceed limit and batch isn't empty, start new batch
        if current_tokens + node_tokens > target_tokens && !current_batch.is_empty() {
            batches.push(current_batch);
            current_batch = Vec::new();
            current_tokens = 0;
        }

        current_batch.push(*node);
        current_tokens += node_tokens;
    }

    // Add final batch
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    batches
}

/// Extract a field from response text
fn extract_field(text: &str, field_name: &str) -> String {
    text.lines()
        .find(|line| line.starts_with(field_name))
        .and_then(|line| line.strip_prefix(field_name))
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Extract a list from response text
fn extract_list(text: &str, field_name: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut in_list = false;

    for line in text.lines() {
        if line.starts_with(field_name) {
            in_list = true;
            continue;
        }

        if in_list {
            let trimmed = line.trim();
            if trimmed.starts_with('-') || trimmed.starts_with('•') {
                items.push(
                    trimmed
                        .trim_start_matches('-')
                        .trim_start_matches('•')
                        .trim()
                        .to_string(),
                );
            } else if !trimmed.is_empty() && !trimmed.contains(':') {
                items.push(trimmed.to_string());
            } else if trimmed.contains(':') && items.len() > 0 {
                // Hit next section
                break;
            }
        }
    }

    items
}

/// Save generation results to file
pub fn save_documentation(
    result: &GenerationResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(output_path, json)?;
    Ok(())
}
