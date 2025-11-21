// graph/mod.rs
// Phase 3: The Docpack Graph - semantic universe binding

pub mod builder;
pub mod resolver;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Unique identifier for graph nodes
pub type NodeId = String;

/// The core graph structure that represents the entire codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocpackGraph {
    /// All nodes in the graph
    pub nodes: HashMap<NodeId, Node>,

    /// All edges in the graph
    pub edges: Vec<Edge>,

    /// Metadata about the graph
    pub metadata: GraphMetadata,
}

/// Metadata about the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub repository_name: Option<String>,
    pub total_files: usize,
    pub total_symbols: usize,
    pub languages: HashSet<String>,
    pub created_at: String,
}

/// Node in the graph - represents a semantic entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub location: Location,
    pub metadata: NodeMetadata,
}

/// Types of nodes in the graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Function(FunctionNode),
    Type(TypeNode),
    Trait(TraitNode),
    Module(ModuleNode),
    Constant(ConstantNode),
    File(FileNode),
    Cluster(ClusterNode),
    Package(PackageNode),
    Macro(MacroNode),
}

/// Function/method node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FunctionNode {
    pub name: String,
    pub signature: String,
    pub is_public: bool,
    pub is_async: bool,
    pub is_method: bool,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
}

/// Type node (struct, class, enum, interface)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TypeNode {
    pub name: String,
    pub kind: TypeKind,
    pub is_public: bool,
    pub fields: Vec<Field>,
    pub methods: Vec<String>, // NodeIds of method functions
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Struct,
    Class,
    Enum,
    Interface,
    Trait,
    Union,
    TypeAlias,
}

/// Trait/interface node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TraitNode {
    pub name: String,
    pub is_public: bool,
    pub methods: Vec<String>,      // Method signatures
    pub implementors: Vec<String>, // NodeIds of implementing types
}

/// Module/namespace node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModuleNode {
    pub name: String,
    pub path: String,
    pub is_public: bool,
    pub children: Vec<NodeId>,
}

/// Constant/static node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ConstantNode {
    pub name: String,
    pub value_type: Option<String>,
    pub is_public: bool,
}

/// File node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct FileNode {
    pub path: String,
    pub language: String,
    pub size_bytes: usize,
    pub line_count: usize,
    pub symbols: Vec<NodeId>,
}

/// Semantic cluster node (from embedding analysis)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub name: String,
    pub topic: Option<String>,
    pub members: Vec<NodeId>,
    pub keywords: Vec<String>,
    pub centroid: Option<Vec<f32>>,
}

impl PartialEq for ClusterNode {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.topic == other.topic && self.members == other.members
    }
}

impl Eq for ClusterNode {}

impl std::hash::Hash for ClusterNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.topic.hash(state);
        self.members.hash(state);
    }
}

/// Embeddable content extracted from nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddableContent {
    pub node_id: NodeId,
    pub text: String,
    pub content_type: ContentType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContentType {
    FunctionBody,
    Docstring,
    TypeDefinition,
    ModuleContent,
    Comment,
}

/// Embedding vector for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub node_id: NodeId,
    pub vector: Vec<f32>,
    pub model: String,
}

/// Package/crate node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PackageNode {
    pub name: String,
    pub version: Option<String>,
    pub modules: Vec<NodeId>,
}

/// Macro node (declarative and procedural)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MacroNode {
    pub name: String,
    pub is_public: bool,
    pub macro_type: MacroType,
    pub pattern: Option<String>, // For declarative macros
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MacroType {
    Declarative, // macro_rules!
    Procedural,  // #[proc_macro]
    Derive,      // #[proc_macro_derive]
    Attribute,   // #[proc_macro_attribute]
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Parameter {
    pub name: String,
    pub param_type: Option<String>,
    pub is_mutable: bool,
}

/// Type field
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Field {
    pub name: String,
    pub field_type: Option<String>,
    pub is_public: bool,
}

/// Location in source code
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Location {
    pub file: String,
    pub start_line: usize,
    pub end_line: usize,
    pub start_col: usize,
    pub end_col: usize,
}

/// Node metadata (complexity, metrics, etc.)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeMetadata {
    pub complexity: Option<u32>,
    pub fan_in: usize,
    pub fan_out: usize,
    pub is_public_api: bool,
    pub docstring: Option<String>,
    pub tags: Vec<String>,
    /// Source code snippet for this node (optional, to save space)
    pub source_snippet: Option<String>,
}

/// Edge in the graph - represents a relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
}

/// Types of edges in the graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    /// Function A calls function B
    Calls,

    /// Module A imports module B
    Imports,

    /// Type A references type B
    TypeReference,

    /// Data flows from A to B
    DataFlow,

    /// Module A owns symbol B
    ModuleOwnership,

    /// Type A implements trait B
    TraitImplementation,

    /// Type A extends/inherits type B
    Inheritance,

    /// Function A is a method of type B
    MethodOf,

    /// Symbol A is defined in file B
    DefinedIn,

    /// Variable/binding has inferred type (e.g., let x = foo() -> x has type Foo)
    InferredType,

    /// Method call resolved to specific trait implementation
    TraitMethodCall,

    /// Method call dispatched to specific implementation
    MethodDispatch,

    /// Macro invocation expands to code
    MacroExpansion,

    /// Trait provides method (trait -> method definition)
    TraitProvides,
}

impl DocpackGraph {
    /// Create a new empty graph
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            metadata: GraphMetadata {
                repository_name: None,
                total_files: 0,
                total_symbols: 0,
                languages: HashSet::new(),
                created_at: chrono::Utc::now().to_rfc3339(),
            },
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Get a node by ID
    #[allow(dead_code)]
    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get all edges from a node
    pub fn get_outgoing_edges(&self, source: &NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| &e.source == source).collect()
    }

    /// Get all edges to a node
    pub fn get_incoming_edges(&self, target: &NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| &e.target == target).collect()
    }

    /// Get all edges of a specific kind
    #[allow(dead_code)]
    pub fn get_edges_by_kind(&self, kind: EdgeKind) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.kind == kind).collect()
    }

    /// Calculate fan-in for a node (how many things depend on it)
    pub fn calculate_fan_in(&self, node_id: &NodeId) -> usize {
        self.get_incoming_edges(node_id).len()
    }

    /// Calculate fan-out for a node (how many things it depends on)
    pub fn calculate_fan_out(&self, node_id: &NodeId) -> usize {
        self.get_outgoing_edges(node_id).len()
    }

    /// Get all nodes of a specific kind
    pub fn get_nodes_by_kind(&self, predicate: impl Fn(&NodeKind) -> bool) -> Vec<&Node> {
        self.nodes.values().filter(|n| predicate(&n.kind)).collect()
    }

    /// Update metadata counts
    pub fn update_metadata(&mut self) {
        self.metadata.total_symbols = self.nodes.len();
        self.metadata.total_files = self
            .nodes
            .values()
            .filter(|n| matches!(n.kind, NodeKind::File(_)))
            .count();

        self.metadata.languages = self
            .nodes
            .values()
            .filter_map(|n| {
                if let NodeKind::File(f) = &n.kind {
                    Some(f.language.clone())
                } else {
                    None
                }
            })
            .collect();
    }
}

impl Default for DocpackGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Node {
    /// Create a new node
    pub fn new(id: NodeId, kind: NodeKind, location: Location) -> Self {
        Self {
            id,
            kind,
            location,
            metadata: NodeMetadata::default(),
        }
    }

    /// Get the node's name
    pub fn name(&self) -> String {
        match &self.kind {
            NodeKind::Function(f) => f.name.clone(),
            NodeKind::Type(t) => t.name.clone(),
            NodeKind::Trait(t) => t.name.clone(),
            NodeKind::Module(m) => m.name.clone(),
            NodeKind::Constant(c) => c.name.clone(),
            NodeKind::File(f) => f.path.clone(),
            NodeKind::Cluster(c) => c.name.clone(),
            NodeKind::Package(p) => p.name.clone(),
            NodeKind::Macro(m) => m.name.clone(),
        }
    }

    /// Check if node is public
    pub fn is_public(&self) -> bool {
        match &self.kind {
            NodeKind::Function(f) => f.is_public,
            NodeKind::Type(t) => t.is_public,
            NodeKind::Trait(t) => t.is_public,
            NodeKind::Module(m) => m.is_public,
            NodeKind::Constant(c) => c.is_public,
            NodeKind::Macro(m) => m.is_public,
            NodeKind::File(_) | NodeKind::Cluster(_) | NodeKind::Package(_) => true,
        }
    }
}

impl Edge {
    /// Create a new edge
    pub fn new(source: NodeId, target: NodeId, kind: EdgeKind) -> Self {
        Self {
            source,
            target,
            kind,
        }
    }
}

// Helper function to generate node IDs
pub fn generate_node_id(file: &str, symbol_type: &str, name: &str) -> NodeId {
    format!("{}::{}::{}", file, symbol_type, name)
}

impl DocpackGraph {
    /// Serialize graph to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize graph from JSON
    #[allow(dead_code)]
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Save graph to file
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load graph from file
    #[allow(dead_code)]
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&json)?)
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        let functions = self
            .get_nodes_by_kind(|k| matches!(k, NodeKind::Function(_)))
            .len();
        let types = self
            .get_nodes_by_kind(|k| matches!(k, NodeKind::Type(_)))
            .len();
        let modules = self
            .get_nodes_by_kind(|k| matches!(k, NodeKind::Module(_)))
            .len();
        let files = self
            .get_nodes_by_kind(|k| matches!(k, NodeKind::File(_)))
            .len();

        GraphStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            functions,
            types,
            modules,
            files,
            languages: self.metadata.languages.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub functions: usize,
    pub types: usize,
    pub modules: usize,
    pub files: usize,
    pub languages: usize,
}
