// graph/resolver.rs
// Reference resolution engine - creates edges between symbols by resolving references

use super::{DocpackGraph, Edge, EdgeKind, Node, NodeId, NodeKind, PackageNode};
use crate::pipeline::parse::ParsedFile;
use std::collections::HashMap;
use tree_sitter::Node as TSNode;

/// Context for resolving references during AST traversal
#[derive(Debug, Clone)]
pub struct ResolutionContext {
    /// Current file being processed
    pub current_file: String,
    /// Current module path (e.g., ["crate", "module", "submodule"])
    #[allow(dead_code)] // kept for future use during more detailed module resolution
    pub module_path: Vec<String>,
    /// Current function scope (None if not in a function)
    pub current_function: Option<NodeId>,
    /// Current type scope (None if not in a type)
    pub current_type: Option<NodeId>,
    /// Language of the current file
    pub language: String,
}

/// Reference resolver - finds and resolves all references in code
pub struct ReferenceResolver {
    /// The graph we're adding edges to
    graph: DocpackGraph,

    /// Symbol name -> NodeId mapping for quick lookups
    symbol_registry: HashMap<String, NodeId>,

    /// Module path -> NodeId mapping
    module_registry: HashMap<String, NodeId>,

    /// Type name -> NodeId mapping
    type_registry: HashMap<String, NodeId>,

    /// File path -> NodeId mapping
    file_registry: HashMap<String, NodeId>,

    /// External package/library -> NodeId mapping
    package_registry: HashMap<String, NodeId>,

    /// Trait name -> NodeId mapping
    trait_registry: HashMap<String, NodeId>,

    /// Macro name -> NodeId mapping
    macro_registry: HashMap<String, NodeId>,

    /// Track pending references to resolve in second pass
    pending_calls: Vec<PendingReference>,
    pending_imports: Vec<PendingReference>,
    pending_type_refs: Vec<PendingReference>,
    pending_returns: Vec<(NodeId, String)>, // function_id, return_type

    /// Track type inference: variable binding -> inferred type
    pending_type_inference: Vec<TypeInferenceInfo>,

    /// Track trait implementations: type -> trait
    pending_trait_impls: Vec<TraitImplementationInfo>,

    /// Track method calls for dispatch resolution
    pending_method_calls: Vec<MethodCallInfo>,

    /// Track macro invocations
    pending_macro_calls: Vec<MacroInvocationInfo>,

    /// Track trait method definitions
    trait_methods: HashMap<String, Vec<String>>, // trait_name -> method_names
}

/// A reference that needs to be resolved
#[derive(Debug, Clone)]
pub struct PendingReference {
    pub source_id: NodeId,
    pub target_name: String,
    #[allow(dead_code)] // context currently carried for potential future use
    pub context: ResolutionContext,
}

/// Type inference information for variable bindings
#[derive(Debug, Clone)]
pub struct TypeInferenceInfo {
    pub binding_name: String,
    pub inferred_from_call: Option<String>, // function name that returns the type
    pub explicit_type: Option<String>,      // explicit type annotation if present
    pub context: ResolutionContext,
}

/// Trait implementation information
#[derive(Debug, Clone)]
pub struct TraitImplementationInfo {
    pub type_name: String,
    pub trait_name: String,
    pub context: ResolutionContext,
}

/// Method call information for dispatch resolution
#[derive(Debug, Clone)]
pub struct MethodCallInfo {
    pub caller_id: Option<NodeId>,
    pub receiver_type: Option<String>, // Type of the object receiving the method call
    pub method_name: String,
    pub context: ResolutionContext,
}

/// Macro invocation information
#[derive(Debug, Clone)]
pub struct MacroInvocationInfo {
    pub caller_id: Option<NodeId>,
    pub macro_name: String,
    pub context: ResolutionContext,
}

impl ReferenceResolver {
    /// Create a new resolver with an existing graph
    pub fn new(graph: DocpackGraph) -> Self {
        let mut resolver = Self {
            graph,
            symbol_registry: HashMap::new(),
            module_registry: HashMap::new(),
            type_registry: HashMap::new(),
            file_registry: HashMap::new(),
            package_registry: HashMap::new(),
            trait_registry: HashMap::new(),
            macro_registry: HashMap::new(),
            pending_calls: Vec::new(),
            pending_imports: Vec::new(),
            pending_type_refs: Vec::new(),
            pending_returns: Vec::new(),
            pending_type_inference: Vec::new(),
            pending_trait_impls: Vec::new(),
            pending_method_calls: Vec::new(),
            pending_macro_calls: Vec::new(),
            trait_methods: HashMap::new(),
        };

        // Build registries from existing graph nodes
        resolver.build_registries();

        resolver
    }

    /// Build lookup registries from existing graph nodes
    fn build_registries(&mut self) {
        for (id, node) in &self.graph.nodes {
            match &node.kind {
                NodeKind::Function(func) => {
                    self.symbol_registry.insert(func.name.clone(), id.clone());

                    // Store return type for later resolution
                    if let Some(return_type) = &func.return_type {
                        self.pending_returns.push((id.clone(), return_type.clone()));
                    }
                }
                NodeKind::Type(type_node) => {
                    self.symbol_registry
                        .insert(type_node.name.clone(), id.clone());
                    self.type_registry
                        .insert(type_node.name.clone(), id.clone());
                }
                NodeKind::Trait(trait_node) => {
                    self.symbol_registry.insert(trait_node.name.clone(), id.clone());
                    self.trait_registry.insert(trait_node.name.clone(), id.clone());
                    // Store trait methods for later resolution
                    self.trait_methods.insert(trait_node.name.clone(), trait_node.methods.clone());
                }
                NodeKind::Module(module) => {
                    self.module_registry.insert(module.path.clone(), id.clone());
                    self.symbol_registry.insert(module.name.clone(), id.clone());
                }
                NodeKind::File(file) => {
                    self.file_registry.insert(file.path.clone(), id.clone());
                }
                NodeKind::Package(pkg) => {
                    self.package_registry.insert(pkg.name.clone(), id.clone());
                }
                NodeKind::Constant(constant) => {
                    self.symbol_registry
                        .insert(constant.name.clone(), id.clone());
                }
                NodeKind::Macro(macro_node) => {
                    self.symbol_registry.insert(macro_node.name.clone(), id.clone());
                    self.macro_registry.insert(macro_node.name.clone(), id.clone());
                }
                _ => {}
            }
        }
    }

    /// Resolve all references in parsed files and add edges
    pub fn resolve_references(mut self, parsed_files: &[ParsedFile]) -> DocpackGraph {
        println!("   🔗 Resolving references...");

        // First pass: Traverse AST and collect all references
        for parsed_file in parsed_files {
            self.traverse_for_references(parsed_file);
        }

        println!("      • Found {} function calls", self.pending_calls.len());
        println!("      • Found {} imports", self.pending_imports.len());
        println!(
            "      • Found {} type references",
            self.pending_type_refs.len()
        );
        println!("      • Found {} return types", self.pending_returns.len());
        println!("      • Found {} type inferences", self.pending_type_inference.len());
        println!("      • Found {} trait implementations", self.pending_trait_impls.len());
        println!("      • Found {} method calls", self.pending_method_calls.len());
        println!("      • Found {} macro invocations", self.pending_macro_calls.len());

        // Second pass: Resolve all pending references
        self.resolve_function_calls();
        self.resolve_module_imports();
        self.resolve_type_references();
        self.resolve_return_type_edges();
        self.resolve_file_symbol_edges();
        self.resolve_external_library_edges();

        // Third pass: Enhanced resolution
        self.resolve_type_inference();
        self.resolve_trait_implementations();
        self.resolve_method_dispatch();
        self.resolve_macro_expansions();

        println!("      ✓ Created {} edges", self.graph.edges.len());

        self.graph
    }

    /// Traverse AST to find all references
    fn traverse_for_references(&mut self, parsed_file: &ParsedFile) {
        let context = ResolutionContext {
            current_file: parsed_file.filename.clone(),
            module_path: vec!["crate".to_string()],
            current_function: None,
            current_type: None,
            language: parsed_file.language.clone(),
        };

        let root = parsed_file.tree.root_node();
        self.traverse_node(&root, &parsed_file.source, context);
    }

    /// Recursively traverse AST nodes
    fn traverse_node(&mut self, node: &TSNode, source: &[u8], mut context: ResolutionContext) {
        match context.language.as_str() {
            "rust" => self.traverse_rust_node(node, source, &mut context),
            "python" => self.traverse_python_node(node, source, &mut context),
            "javascript" | "typescript" | "tsx" | "jsx" => {
                self.traverse_js_ts_node(node, source, &mut context)
            }
            _ => {
                // Generic traversal
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.traverse_node(&child, source, context.clone());
                }
            }
        }
    }

    /// Traverse Rust-specific nodes
    fn traverse_rust_node(
        &mut self,
        node: &TSNode,
        source: &[u8],
        context: &mut ResolutionContext,
    ) {
        let kind = node.kind();

        match kind {
            // Track function scope
            "function_item" => {
                if let Some(name) = self.find_child_text(node, "identifier", source) {
                    if let Some(func_id) = self.symbol_registry.get(&name) {
                        context.current_function = Some(func_id.clone());
                    }
                }
            }

            // Track impl block scope (for methods and trait implementations)
            "impl_item" => {
                self.extract_rust_impl_block(node, source, context);
                if let Some(type_name) = self.find_child_text(node, "type_identifier", source) {
                    if let Some(type_id) = self.type_registry.get(&type_name) {
                        context.current_type = Some(type_id.clone());
                    }
                }
            }

            // Function call
            "call_expression" => {
                self.extract_rust_call(node, source, context);
            }

            // Method call (field_expression with call)
            "field_expression" => {
                // Check if parent is a call_expression to detect method calls
                if let Some(parent) = node.parent() {
                    if parent.kind() == "call_expression" {
                        self.extract_rust_method_call(node, source, context);
                    }
                }
            }

            // Use statement (import)
            "use_declaration" => {
                self.extract_rust_import(node, source, context);
            }

            // Type annotation
            "type_identifier" | "generic_type" => {
                self.extract_rust_type_ref(node, source, context);
            }

            // Let binding with type inference
            "let_declaration" => {
                self.extract_rust_type_inference(node, source, context);
            }

            // Macro invocation
            "macro_invocation" => {
                self.extract_rust_macro_call(node, source, context);
            }

            _ => {}
        }

        // Recursively process children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.traverse_rust_node(&child, source, context);
        }
    }

    /// Extract function call from Rust AST
    fn extract_rust_call(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(source_id) = &context.current_function {
            // Get the function being called
            if let Some(function_node) = node.child_by_field_name("function") {
                if let Some(callee_name) = self.get_node_text(&function_node, source) {
                    // Handle simple function names and method calls
                    let function_name = callee_name
                        .split("::")
                        .last()
                        .or_else(|| callee_name.split('.').last())
                        .unwrap_or(&callee_name)
                        .to_string();

                    self.pending_calls.push(PendingReference {
                        source_id: source_id.clone(),
                        target_name: function_name,
                        context: context.clone(),
                    });
                }
            }
        }
    }

    /// Extract import/use statement from Rust AST
    fn extract_rust_import(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        // Get the imported path
        if let Some(import_text) = self.get_node_text(node, source) {
            // Parse "use std::collections::HashMap;" -> ["std", "collections", "HashMap"]
            let parts: Vec<&str> = import_text
                .trim_start_matches("use ")
                .trim_end_matches(';')
                .split("::")
                .collect();

            if let Some(&first_part) = parts.first() {
                // Check if it's an external crate or internal module
                let is_external =
                    first_part != "crate" && first_part != "self" && first_part != "super";

                if is_external {
                    // Track as external library import
                    let file_id = self.file_registry.get(&context.current_file).cloned();
                    if let Some(source_id) = file_id {
                        self.pending_imports.push(PendingReference {
                            source_id,
                            target_name: first_part.to_string(),
                            context: context.clone(),
                        });
                    }
                } else if let Some(&module_name) = parts.last() {
                    // Track as internal module import
                    let file_id = self.file_registry.get(&context.current_file).cloned();
                    if let Some(source_id) = file_id {
                        self.pending_imports.push(PendingReference {
                            source_id,
                            target_name: module_name.to_string(),
                            context: context.clone(),
                        });
                    }
                }
            }
        }
    }

    /// Extract type reference from Rust AST
    fn extract_rust_type_ref(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(type_name) = self.get_node_text(node, source) {
            // Clean up generic types: Vec<String> -> Vec
            let base_type = type_name
                .split('<')
                .next()
                .unwrap_or(&type_name)
                .to_string();

            // Only track user-defined types (not primitives)
            if !self.is_primitive_type(&base_type) {
                if let Some(source_id) = context
                    .current_function
                    .as_ref()
                    .or(context.current_type.as_ref())
                {
                    self.pending_type_refs.push(PendingReference {
                        source_id: source_id.clone(),
                        target_name: base_type,
                        context: context.clone(),
                    });
                }
            }
        }
    }

    /// Extract type inference from let bindings
    fn extract_rust_type_inference(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        // Pattern: let binding_name: Type = value  OR  let binding_name = function_call()
        let mut binding_name = None;
        let mut explicit_type = None;
        let mut inferred_from_call = None;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    if binding_name.is_none() {
                        binding_name = self.get_node_text(&child, source);
                    }
                }
                "type_identifier" | "generic_type" => {
                    explicit_type = self.get_node_text(&child, source);
                }
                "call_expression" => {
                    // Try to extract the function being called
                    if let Some(func_node) = child.child_by_field_name("function") {
                        if let Some(func_name) = self.get_node_text(&func_node, source) {
                            let simple_name = func_name
                                .split("::")
                                .last()
                                .or_else(|| func_name.split('.').last())
                                .unwrap_or(&func_name)
                                .to_string();
                            inferred_from_call = Some(simple_name);
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(name) = binding_name {
            self.pending_type_inference.push(TypeInferenceInfo {
                binding_name: name,
                inferred_from_call,
                explicit_type,
                context: context.clone(),
            });
        }
    }

    /// Extract trait implementation from impl blocks
    fn extract_rust_impl_block(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        // Pattern: impl TraitName for TypeName
        let mut type_name = None;
        let mut trait_name = None;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "type_identifier" => {
                    if type_name.is_none() {
                        type_name = self.get_node_text(&child, source);
                    } else if trait_name.is_none() {
                        // The second type_identifier is the trait in "impl Trait for Type"
                        trait_name = self.get_node_text(&child, source);
                    }
                }
                _ => {}
            }
        }

        // Check if this is "impl Trait for Type" pattern
        if let Some(text) = self.get_node_text(node, source) {
            if text.contains(" for ") {
                // Parse "impl Trait for Type" - swap them
                if let (Some(t), Some(tr)) = (type_name.as_ref(), trait_name.as_ref()) {
                    self.pending_trait_impls.push(TraitImplementationInfo {
                        type_name: t.clone(),
                        trait_name: tr.clone(),
                        context: context.clone(),
                    });
                }
            }
        }
    }

    /// Extract method call for dispatch resolution
    fn extract_rust_method_call(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        // Pattern: receiver.method_name()
        let mut receiver_type = None;
        let mut method_name = None;

        if let Some(value_node) = node.child_by_field_name("value") {
            // Try to get the type of the receiver (simplified - could be enhanced)
            receiver_type = self.get_node_text(&value_node, source);
        }

        if let Some(field_node) = node.child_by_field_name("field") {
            method_name = self.get_node_text(&field_node, source);
        }

        if let Some(method) = method_name {
            self.pending_method_calls.push(MethodCallInfo {
                caller_id: context.current_function.clone(),
                receiver_type,
                method_name: method,
                context: context.clone(),
            });
        }
    }

    /// Extract macro invocation
    fn extract_rust_macro_call(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        // Pattern: macro_name!(args)
        if let Some(macro_node) = node.child_by_field_name("macro") {
            if let Some(mut macro_name) = self.get_node_text(&macro_node, source) {
                // Remove trailing ! if present
                if macro_name.ends_with('!') {
                    macro_name = macro_name[..macro_name.len() - 1].to_string();
                }

                self.pending_macro_calls.push(MacroInvocationInfo {
                    caller_id: context.current_function.clone(),
                    macro_name,
                    context: context.clone(),
                });
            }
        }
    }

    /// Traverse Python-specific nodes
    fn traverse_python_node(
        &mut self,
        node: &TSNode,
        source: &[u8],
        context: &mut ResolutionContext,
    ) {
        let kind = node.kind();

        match kind {
            "function_definition" => {
                if let Some(name) = self.find_child_text(node, "identifier", source) {
                    if let Some(func_id) = self.symbol_registry.get(&name) {
                        context.current_function = Some(func_id.clone());
                    }
                }
            }

            "class_definition" => {
                if let Some(name) = self.find_child_text(node, "identifier", source) {
                    if let Some(type_id) = self.type_registry.get(&name) {
                        context.current_type = Some(type_id.clone());
                    }
                }
            }

            "call" => {
                self.extract_python_call(node, source, context);
            }

            "import_statement" | "import_from_statement" => {
                self.extract_python_import(node, source, context);
            }

            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.traverse_python_node(&child, source, context);
        }
    }

    /// Extract function call from Python AST
    fn extract_python_call(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(source_id) = &context.current_function {
            if let Some(function_node) = node.child_by_field_name("function") {
                if let Some(callee_name) = self.get_node_text(&function_node, source) {
                    let function_name = callee_name
                        .split('.')
                        .last()
                        .unwrap_or(&callee_name)
                        .to_string();

                    self.pending_calls.push(PendingReference {
                        source_id: source_id.clone(),
                        target_name: function_name,
                        context: context.clone(),
                    });
                }
            }
        }
    }

    /// Extract import from Python AST
    fn extract_python_import(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(import_text) = self.get_node_text(node, source) {
            // Parse "import numpy" or "from sklearn import datasets"
            let parts: Vec<&str> = import_text.split_whitespace().collect();

            let package_name = if parts.len() > 1 {
                if parts[0] == "from" {
                    parts[1].to_string()
                } else {
                    parts[1].to_string()
                }
            } else {
                return;
            };

            let file_id = self.file_registry.get(&context.current_file).cloned();
            if let Some(source_id) = file_id {
                self.pending_imports.push(PendingReference {
                    source_id,
                    target_name: package_name,
                    context: context.clone(),
                });
            }
        }
    }

    /// Traverse JavaScript/TypeScript nodes
    fn traverse_js_ts_node(
        &mut self,
        node: &TSNode,
        source: &[u8],
        context: &mut ResolutionContext,
    ) {
        let kind = node.kind();

        match kind {
            "function_declaration" | "method_definition" | "arrow_function" => {
                if let Some(name) = self
                    .find_child_text(node, "identifier", source)
                    .or_else(|| self.find_child_text(node, "property_identifier", source))
                {
                    if let Some(func_id) = self.symbol_registry.get(&name) {
                        context.current_function = Some(func_id.clone());
                    }
                }
            }

            "class_declaration" => {
                if let Some(name) = self.find_child_text(node, "identifier", source) {
                    if let Some(type_id) = self.type_registry.get(&name) {
                        context.current_type = Some(type_id.clone());
                    }
                }
            }

            "call_expression" => {
                self.extract_js_call(node, source, context);
            }

            "import_statement" => {
                self.extract_js_import(node, source, context);
            }

            _ => {}
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.traverse_js_ts_node(&child, source, context);
        }
    }

    /// Extract function call from JS/TS AST
    fn extract_js_call(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(source_id) = &context.current_function {
            if let Some(function_node) = node.child_by_field_name("function") {
                if let Some(callee_name) = self.get_node_text(&function_node, source) {
                    let function_name = callee_name
                        .split('.')
                        .last()
                        .unwrap_or(&callee_name)
                        .to_string();

                    self.pending_calls.push(PendingReference {
                        source_id: source_id.clone(),
                        target_name: function_name,
                        context: context.clone(),
                    });
                }
            }
        }
    }

    /// Extract import from JS/TS AST
    fn extract_js_import(&mut self, node: &TSNode, source: &[u8], context: &ResolutionContext) {
        if let Some(import_text) = self.get_node_text(node, source) {
            // Parse: import { something } from 'package' or import React from 'react'
            if let Some(from_idx) = import_text.find("from") {
                let package_part = &import_text[from_idx + 4..].trim();
                let package_name = package_part
                    .trim_matches(|c| c == '\'' || c == '"' || c == ';')
                    .trim()
                    .to_string();

                // Check if it's a relative import (starts with ./ or ../)
                let is_external = !package_name.starts_with('.');

                if is_external {
                    let file_id = self.file_registry.get(&context.current_file).cloned();
                    if let Some(source_id) = file_id {
                        self.pending_imports.push(PendingReference {
                            source_id,
                            target_name: package_name,
                            context: context.clone(),
                        });
                    }
                }
            }
        }
    }

    /// Resolve function call edges: function → function calls
    fn resolve_function_calls(&mut self) {
        let mut resolved = 0;
        for pending in &self.pending_calls {
            if let Some(target_id) = self.symbol_registry.get(&pending.target_name) {
                self.graph.add_edge(Edge::new(
                    pending.source_id.clone(),
                    target_id.clone(),
                    EdgeKind::Calls,
                ));
                resolved += 1;
            }
        }
        println!(
            "      • Resolved {} / {} function calls",
            resolved,
            self.pending_calls.len()
        );
    }

    /// Resolve module import edges: module → module import
    fn resolve_module_imports(&mut self) {
        let mut resolved = 0;
        for pending in &self.pending_imports {
            // Try to find as module first
            if let Some(target_id) = self.module_registry.get(&pending.target_name) {
                self.graph.add_edge(Edge::new(
                    pending.source_id.clone(),
                    target_id.clone(),
                    EdgeKind::Imports,
                ));
                resolved += 1;
            } else if let Some(target_id) = self.package_registry.get(&pending.target_name) {
                // Or as external package
                self.graph.add_edge(Edge::new(
                    pending.source_id.clone(),
                    target_id.clone(),
                    EdgeKind::Imports,
                ));
                resolved += 1;
            } else {
                // Create a package node for unknown external libraries
                let pkg_id = format!("package::{}", pending.target_name);
                if !self.graph.nodes.contains_key(&pkg_id) {
                    let pkg_node = Node::new(
                        pkg_id.clone(),
                        NodeKind::Package(PackageNode {
                            name: pending.target_name.clone(),
                            version: None,
                            modules: Vec::new(),
                        }),
                        super::Location {
                            file: "external".to_string(),
                            start_line: 0,
                            end_line: 0,
                            start_col: 0,
                            end_col: 0,
                        },
                    );
                    self.graph.add_node(pkg_node);
                    self.package_registry
                        .insert(pending.target_name.clone(), pkg_id.clone());
                }

                self.graph.add_edge(Edge::new(
                    pending.source_id.clone(),
                    pkg_id,
                    EdgeKind::Imports,
                ));
                resolved += 1;
            }
        }
        println!(
            "      • Resolved {} / {} imports",
            resolved,
            self.pending_imports.len()
        );
    }

    /// Resolve type reference edges: type → functions returning that type
    fn resolve_type_references(&mut self) {
        let mut resolved = 0;
        for pending in &self.pending_type_refs {
            if let Some(target_id) = self.type_registry.get(&pending.target_name) {
                self.graph.add_edge(Edge::new(
                    pending.source_id.clone(),
                    target_id.clone(),
                    EdgeKind::TypeReference,
                ));
                resolved += 1;
            }
        }
        println!(
            "      • Resolved {} / {} type references",
            resolved,
            self.pending_type_refs.len()
        );
    }

    /// Create edges from types to functions that return them
    fn resolve_return_type_edges(&mut self) {
        let mut created = 0;
        for (func_id, return_type) in &self.pending_returns {
            // Clean up return type
            let base_type = return_type
                .trim()
                .split('<')
                .next()
                .unwrap_or(return_type)
                .split('[')
                .next()
                .unwrap_or(return_type)
                .trim()
                .to_string();

            if !self.is_primitive_type(&base_type) {
                if let Some(type_id) = self.type_registry.get(&base_type) {
                    // Create edge from type to function (type is returned by function)
                    self.graph.add_edge(Edge::new(
                        type_id.clone(),
                        func_id.clone(),
                        EdgeKind::TypeReference,
                    ));
                    created += 1;
                }
            }
        }
        println!("      • Created {} return type edges", created);
    }

    /// Create edges from files to symbols defined in them
    fn resolve_file_symbol_edges(&mut self) {
        let mut created = 0;
        let mut edges_to_add = Vec::new();

        for (_file_path, file_id) in &self.file_registry {
            if let Some(file_node) = self.graph.nodes.get(file_id) {
                if let NodeKind::File(f) = &file_node.kind {
                    for symbol_id in &f.symbols {
                        edges_to_add.push((symbol_id.clone(), file_id.clone()));
                    }
                }
            }
        }

        for (symbol_id, file_id) in edges_to_add {
            self.graph
                .add_edge(Edge::new(symbol_id, file_id, EdgeKind::DefinedIn));
            created += 1;
        }

        println!("      • Created {} file symbol edges", created);
    }

    /// Create edges from external libraries to their usage sites
    fn resolve_external_library_edges(&mut self) {
        // This is already handled in resolve_module_imports
        // The edges go from files/modules to external packages
        let external_package_count = self.package_registry.len();
        println!(
            "      • Tracked {} external packages",
            external_package_count
        );
    }

    /// Resolve type inference edges
    fn resolve_type_inference(&mut self) {
        let mut resolved = 0;

        for inference in &self.pending_type_inference {
            // If explicit type annotation exists, create edge to that type
            if let Some(ref type_name) = inference.explicit_type {
                let base_type = type_name.split('<').next().unwrap_or(type_name);
                if let Some(type_id) = self.type_registry.get(base_type) {
                    // We don't have a node for the binding itself, but we can create
                    // an edge from the current function to the type (if in function scope)
                    if let Some(ref caller_id) = inference.context.current_function {
                        self.graph.add_edge(Edge::new(
                            caller_id.clone(),
                            type_id.clone(),
                            EdgeKind::InferredType,
                        ));
                        resolved += 1;
                    }
                }
            }
            // If inferred from function call, look up the function's return type
            else if let Some(ref func_name) = inference.inferred_from_call {
                if let Some(func_id) = self.symbol_registry.get(func_name) {
                    // Get the function's return type
                    if let Some(node) = self.graph.nodes.get(func_id) {
                        if let NodeKind::Function(func_node) = &node.kind {
                            if let Some(ref return_type) = func_node.return_type {
                                let base_type = return_type.split('<').next().unwrap_or(return_type);
                                if let Some(type_id) = self.type_registry.get(base_type) {
                                    if let Some(ref caller_id) = inference.context.current_function {
                                        self.graph.add_edge(Edge::new(
                                            caller_id.clone(),
                                            type_id.clone(),
                                            EdgeKind::InferredType,
                                        ));
                                        resolved += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        println!(
            "      • Resolved {} / {} type inferences",
            resolved,
            self.pending_type_inference.len()
        );
    }

    /// Resolve trait implementation edges
    fn resolve_trait_implementations(&mut self) {
        let mut resolved = 0;

        for trait_impl in &self.pending_trait_impls {
            // Find the type and trait nodes
            if let (Some(type_id), Some(trait_id)) = (
                self.type_registry.get(&trait_impl.type_name),
                self.trait_registry.get(&trait_impl.trait_name),
            ) {
                // Create edge: Type implements Trait
                self.graph.add_edge(Edge::new(
                    type_id.clone(),
                    trait_id.clone(),
                    EdgeKind::TraitImplementation,
                ));
                resolved += 1;
            }
        }

        println!(
            "      • Resolved {} / {} trait implementations",
            resolved,
            self.pending_trait_impls.len()
        );
    }

    /// Resolve method dispatch edges
    fn resolve_method_dispatch(&mut self) {
        let mut resolved = 0;

        for method_call in &self.pending_method_calls {
            // Try to find the method in our symbol registry
            if let Some(method_id) = self.symbol_registry.get(&method_call.method_name) {
                if let Some(ref caller_id) = method_call.caller_id {
                    // Create a MethodDispatch edge from caller to the resolved method
                    self.graph.add_edge(Edge::new(
                        caller_id.clone(),
                        method_id.clone(),
                        EdgeKind::MethodDispatch,
                    ));
                    resolved += 1;
                }
            }

            // Advanced: Try to resolve based on receiver type and trait methods
            if let Some(ref receiver_type) = method_call.receiver_type {
                // Check if this is a known type
                if let Some(type_id) = self.type_registry.get(receiver_type) {
                    // Look through trait implementations to find matching methods
                    for (trait_name, methods) in &self.trait_methods {
                        if methods.contains(&method_call.method_name) {
                            // Check if this type implements this trait
                            if let Some(trait_id) = self.trait_registry.get(trait_name) {
                                // Create a TraitMethodCall edge
                                if let Some(ref caller_id) = method_call.caller_id {
                                    self.graph.add_edge(Edge::new(
                                        caller_id.clone(),
                                        trait_id.clone(),
                                        EdgeKind::TraitMethodCall,
                                    ));
                                    resolved += 1;
                                    break;
                                }
                            }
                        }
                    }

                    // Also check type's own methods
                    if let Some(node) = self.graph.nodes.get(type_id) {
                        if let NodeKind::Type(type_node) = &node.kind {
                            for method_id in &type_node.methods {
                                if let Some(method_node) = self.graph.nodes.get(method_id) {
                                    if let NodeKind::Function(func) = &method_node.kind {
                                        if func.name == method_call.method_name {
                                            if let Some(ref caller_id) = method_call.caller_id {
                                                self.graph.add_edge(Edge::new(
                                                    caller_id.clone(),
                                                    method_id.clone(),
                                                    EdgeKind::MethodDispatch,
                                                ));
                                                resolved += 1;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        println!(
            "      • Resolved {} / {} method dispatches",
            resolved,
            self.pending_method_calls.len()
        );
    }

    /// Resolve macro expansion edges
    fn resolve_macro_expansions(&mut self) {
        let mut resolved = 0;

        for macro_call in &self.pending_macro_calls {
            // Look up the macro in the registry
            if let Some(macro_id) = self.macro_registry.get(&macro_call.macro_name) {
                if let Some(ref caller_id) = macro_call.caller_id {
                    // Create MacroExpansion edge from caller to macro
                    self.graph.add_edge(Edge::new(
                        caller_id.clone(),
                        macro_id.clone(),
                        EdgeKind::MacroExpansion,
                    ));
                    resolved += 1;
                }
            }
            // If macro not found, it might be a standard library macro or external
            // We could optionally create a node for it
        }

        println!(
            "      • Resolved {} / {} macro expansions",
            resolved,
            self.pending_macro_calls.len()
        );
    }

    // Helper methods

    fn get_node_text(&self, node: &TSNode, source: &[u8]) -> Option<String> {
        let text = &source[node.start_byte()..node.end_byte()];
        String::from_utf8(text.to_vec()).ok()
    }

    fn find_child_text(&self, node: &TSNode, child_kind: &str, source: &[u8]) -> Option<String> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == child_kind {
                return self.get_node_text(&child, source);
            }
        }
        None
    }

    fn is_primitive_type(&self, type_name: &str) -> bool {
        matches!(
            type_name,
            // Rust primitives
            "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
            "u8" | "u16" | "u32" | "u64" | "u128" | "usize" |
            "f32" | "f64" | "bool" | "char" | "str" | "String" | "()" | "&str" | "Option" | "Result" | "Vec" |
            // Python primitives
            "int" | "float" | "bytes" | "list" | "dict" | "tuple" | "set" |
            // JavaScript/TypeScript primitives
            "number" | "string" | "boolean" | "any" | "void" | "null" | "undefined"
        )
    }
}
