// Pipeline Phase 4: Complexity & Metrics Analysis
// Computes cyclomatic complexity and other code metrics

use crate::graph::{DocpackGraph, NodeKind};

/// Configuration for analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub calculate_complexity: bool,
    pub detect_public_api: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            calculate_complexity: true,
            detect_public_api: true,
        }
    }
}

/// Result of analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub nodes_analyzed: usize,
    pub complexity_calculated: usize,
    pub public_api_detected: usize,
}

/// Analyze the graph and populate metrics
pub fn analyze_graph(
    graph: &mut DocpackGraph,
    config: &AnalysisConfig,
) -> Result<AnalysisResult, Box<dyn std::error::Error>> {
    let mut nodes_analyzed = 0;
    let mut complexity_calculated = 0;
    let mut public_api_detected = 0;

    // Get all node IDs first to avoid borrow checker issues
    let node_ids: Vec<_> = graph.nodes.keys().cloned().collect();

    for node_id in node_ids {
        nodes_analyzed += 1;

        // Calculate complexity for functions
        if config.calculate_complexity {
            if let Some(node) = graph.nodes.get(&node_id) {
                if matches!(node.kind, NodeKind::Function(_)) {
                    let complexity = calculate_complexity(node);
                    if let Some(node) = graph.nodes.get_mut(&node_id) {
                        node.metadata.complexity = Some(complexity);
                        complexity_calculated += 1;
                    }
                }
            }
        }

        // Detect public API
        if config.detect_public_api {
            if let Some(node) = graph.nodes.get(&node_id) {
                let is_public_api = detect_public_api(node, graph);
                if is_public_api {
                    if let Some(node) = graph.nodes.get_mut(&node_id) {
                        node.metadata.is_public_api = true;
                        public_api_detected += 1;
                    }
                }
            }
        }
    }

    Ok(AnalysisResult {
        nodes_analyzed,
        complexity_calculated,
        public_api_detected,
    })
}

/// Calculate cyclomatic complexity for a function node
/// This is a simplified version based on source code analysis
fn calculate_complexity(node: &crate::graph::Node) -> u32 {
    // Start with base complexity of 1
    let mut complexity = 1u32;

    // If we have source snippet, analyze it for decision points
    if let Some(source) = &node.metadata.source_snippet {
        // Count decision points that increase cyclomatic complexity
        // These are simplified heuristics based on common patterns

        // Conditionals: if, else if, match arms, etc.
        complexity += source.matches("if ").count() as u32;
        complexity += source.matches("else if").count() as u32;
        complexity += source.matches("match ").count() as u32;

        // Count match arms (each arm is a decision point)
        // Simplified: count "=>" which typically indicates match arms
        let arrow_count = source.matches("=>").count() as u32;
        if arrow_count > 0 {
            // Subtract 1 because we already counted the match statement
            complexity += arrow_count.saturating_sub(1);
        }

        // Loops: for, while, loop
        complexity += source.matches("for ").count() as u32;
        complexity += source.matches("while ").count() as u32;
        complexity += source.matches("loop ").count() as u32;

        // Logical operators (&&, ||) add branches
        complexity += source.matches("&&").count() as u32;
        complexity += source.matches("||").count() as u32;

        // Return statements in the middle of functions
        let return_count = source.matches("return").count() as u32;
        if return_count > 1 {
            // Multiple returns add complexity (but not the final one)
            complexity += return_count.saturating_sub(1);
        }

        // Error handling: ?, unwrap, expect
        complexity += source.matches("?").count() as u32;
        complexity += source.matches(".unwrap()").count() as u32;
        complexity += source.matches(".expect(").count() as u32;

        // Try-catch in other languages
        complexity += source.matches("try ").count() as u32;
        complexity += source.matches("catch ").count() as u32;
        complexity += source.matches("except ").count() as u32;
    } else {
        // If no source snippet, use fan-out as a proxy for complexity
        // Functions that call many other functions are likely more complex
        let fan_out = node.metadata.fan_out as u32;
        complexity += (fan_out / 3).min(10); // Cap the contribution
    }

    complexity
}

/// Detect if a node is part of the public API
/// A node is considered public API if:
/// 1. It's marked as public (pub)
/// 2. It's in a public module path
/// 3. It's exported/re-exported
fn detect_public_api(node: &crate::graph::Node, _graph: &DocpackGraph) -> bool {
    // First check: is the node itself public?
    if !node.is_public() {
        return false;
    }

    // Second check: Check based on node type
    match &node.kind {
        NodeKind::Function(f) => {
            // Public functions are API unless they're in private modules
            if !f.is_public {
                return false;
            }
            // If it's a method, it's public API only if the type is public
            if f.is_method {
                // TODO: Could check parent type's visibility here
                return true;
            }
            // Standalone public functions are API
            true
        }
        NodeKind::Type(t) => {
            // Public types are API
            t.is_public
        }
        NodeKind::Trait(t) => {
            // Public traits are API
            t.is_public
        }
        NodeKind::Module(m) => {
            // Public modules that export public items are API
            m.is_public
        }
        NodeKind::Constant(c) => {
            // Public constants are API
            c.is_public
        }
        NodeKind::File(_) | NodeKind::Cluster(_) | NodeKind::Package(_) => {
            // These are not directly part of API
            false
        }
    }
}

/// Analyze and update all metrics in the graph
#[allow(dead_code)]
pub fn update_all_metrics(graph: &mut DocpackGraph) -> Result<(), Box<dyn std::error::Error>> {
    let config = AnalysisConfig::default();
    analyze_graph(graph, &config)?;
    Ok(())
}
