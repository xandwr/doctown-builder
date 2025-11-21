// graph/builder.rs
// Graph builder - constructs the Docpack Graph from parsed AST data

use super::resolver::ReferenceResolver;
use super::{
    ConstantNode, DocpackGraph, Edge, EdgeKind, Field, FileNode, FunctionNode, Location,
    MacroNode, MacroType, ModuleNode, Node, NodeId, NodeKind, Parameter, TraitNode, TypeKind,
    TypeNode, generate_node_id,
};
use crate::pipeline::parse::ParsedFile;
use std::collections::HashMap;
use tree_sitter::Node as TSNode;

pub struct GraphBuilder {
    graph: DocpackGraph,
    // Track current context
    current_file: String,
    current_module: Vec<String>,
    // Track relationships to build edges later
    pending_calls: Vec<(NodeId, String)>, // (caller_id, callee_name)
    pending_type_refs: Vec<(NodeId, String)>, // (user_id, type_name)
    // Symbol resolution maps
    symbol_to_id: HashMap<String, NodeId>,
    // Track file -> symbols mapping
    file_symbols: HashMap<String, Vec<NodeId>>,
    // Track type -> methods mapping for impl blocks
    type_methods: HashMap<NodeId, Vec<NodeId>>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: DocpackGraph::new(),
            current_file: String::new(),
            current_module: Vec::new(),
            pending_calls: Vec::new(),
            pending_type_refs: Vec::new(),
            symbol_to_id: HashMap::new(),
            file_symbols: HashMap::new(),
            type_methods: HashMap::new(),
        }
    }

    /// Build graph from parsed files
    pub fn build(mut self, parsed_files: &[ParsedFile]) -> DocpackGraph {
        // First pass: Extract all nodes (definitions)
        for parsed_file in parsed_files {
            self.process_file(parsed_file);
        }

        // Second pass: Resolve basic relationships and create initial edges
        self.resolve_edges();

        // Third pass: Populate file symbols
        self.populate_file_symbols();

        // Fourth pass: Attach methods to types
        self.attach_methods_to_types();

        // Update metadata
        self.graph.update_metadata();

        // Fifth pass: Use reference resolver for comprehensive edge creation
        let resolver = ReferenceResolver::new(self.graph);
        self.graph = resolver.resolve_references(parsed_files);

        self.graph
    }

    fn process_file(&mut self, parsed_file: &ParsedFile) {
        self.current_file = parsed_file.filename.clone();
        self.current_module.clear();

        // Add file node
        let file_id = generate_node_id(&self.current_file, "file", "");
        let file_node = Node::new(
            file_id.clone(),
            NodeKind::File(FileNode {
                path: parsed_file.filename.clone(),
                language: parsed_file.language.clone(),
                size_bytes: parsed_file.source.len(),
                line_count: parsed_file.source.iter().filter(|&&b| b == b'\n').count() + 1,
                symbols: Vec::new(),
            }),
            Location {
                file: self.current_file.clone(),
                start_line: 1,
                end_line: 0,
                start_col: 0,
                end_col: 0,
            },
        );
        self.graph.add_node(file_node);

        // Parse the AST based on language
        let root = parsed_file.tree.root_node();
        self.process_node(&root, &parsed_file.source, &parsed_file.language);
    }

    fn process_node(&mut self, node: &TSNode, source: &[u8], language: &str) {
        match language {
            "rust" => self.process_rust_node(node, source),
            "python" => self.process_python_node(node, source),
            "javascript" | "typescript" | "tsx" | "jsx" => {
                self.process_js_ts_node(node, source, language)
            }
            "go" => self.process_go_node(node, source),
            "java" => self.process_java_node(node, source),
            "c" | "cpp" => self.process_c_cpp_node(node, source),
            _ => {
                // For unsupported languages, do a generic traversal
                self.generic_traverse(node, source, language);
            }
        }
    }

    fn process_rust_node(&mut self, node: &TSNode, source: &[u8]) {
        let kind = node.kind();

        match kind {
            "function_item" | "function_signature_item" => {
                self.extract_rust_function(node, source);
            }
            "struct_item" => {
                self.extract_rust_struct(node, source);
            }
            "enum_item" => {
                self.extract_rust_enum(node, source);
            }
            "trait_item" => {
                self.extract_rust_trait(node, source);
            }
            "impl_item" => {
                self.extract_rust_impl(node, source);
            }
            "macro_definition" => {
                self.extract_rust_macro(node, source);
            }
            "const_item" | "static_item" => {
                self.extract_rust_constant(node, source);
            }
            "mod_item" => {
                self.extract_rust_module(node, source);
            }
            "call_expression" => {
                // Track function calls for edges
                if let Some(_func_name) = self.get_node_text(node, source) {
                    // We'll need context about which function we're in
                    // This is simplified - real implementation needs scope tracking
                }
            }
            _ => {}
        }

        // Recursively process children
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.process_rust_node(&child, source);
        }
    }

    fn extract_rust_function(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;
        let mut is_async = false;
        let mut parameters = Vec::new();
        let mut return_type = None;

        // Parse function components
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    if self.get_node_text(&child, source) == Some("pub".to_string()) {
                        is_public = true;
                    }
                }
                "async" => {
                    is_async = true;
                }
                "identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                "parameters" => {
                    parameters = self.extract_rust_parameters(&child, source);
                }
                "type" | "return_type" => {
                    return_type = self.get_node_text(&child, source);
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "function", &name);
            let location = self.get_location(node);
            let source_snippet = self.get_node_text(node, source);

            let mut func_node = Node::new(
                node_id.clone(),
                NodeKind::Function(FunctionNode {
                    name: name.clone(),
                    signature: source_snippet.clone().unwrap_or_default(),
                    is_public,
                    is_async,
                    is_method: false, // Will be updated if in impl block
                    parameters,
                    return_type,
                }),
                location,
            );
            func_node.metadata.source_snippet = source_snippet;

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(func_node);
        }
    }

    fn extract_rust_struct(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;
        let mut fields = Vec::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    is_public = true;
                }
                "type_identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                "field_declaration_list" => {
                    fields = self.extract_rust_fields(&child, source);
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "struct", &name);
            let location = self.get_location(node);
            let source_snippet = self.get_node_text(node, source);

            let mut type_node = Node::new(
                node_id.clone(),
                NodeKind::Type(TypeNode {
                    name: name.clone(),
                    kind: TypeKind::Struct,
                    is_public,
                    fields,
                    methods: Vec::new(),
                }),
                location,
            );
            type_node.metadata.source_snippet = source_snippet;

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(type_node);
        }
    }

    fn extract_rust_enum(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    is_public = true;
                }
                "type_identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "enum", &name);
            let location = self.get_location(node);
            let source_snippet = self.get_node_text(node, source);

            let mut type_node = Node::new(
                node_id.clone(),
                NodeKind::Type(TypeNode {
                    name: name.clone(),
                    kind: TypeKind::Enum,
                    is_public,
                    fields: Vec::new(),
                    methods: Vec::new(),
                }),
                location,
            );
            type_node.metadata.source_snippet = source_snippet;

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(type_node);
        }
    }

    fn extract_rust_trait(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;
        let mut methods = Vec::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    is_public = true;
                }
                "type_identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                "declaration_list" => {
                    // Extract trait method signatures
                    let mut method_cursor = child.walk();
                    for method_child in child.children(&mut method_cursor) {
                        if method_child.kind() == "function_signature_item" {
                            if let Some(method_name) = self.find_child_text(&method_child, "identifier", source) {
                                methods.push(method_name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "trait", &name);
            let location = self.get_location(node);

            let trait_node = Node::new(
                node_id.clone(),
                NodeKind::Trait(TraitNode {
                    name: name.clone(),
                    is_public,
                    methods,
                    implementors: Vec::new(),
                }),
                location,
            );

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(trait_node);
        }
    }

    fn extract_rust_impl(&mut self, node: &TSNode, source: &[u8]) {
        // Extract impl blocks and link methods to types
        let mut type_name = None;
        let mut methods = Vec::new();

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "type_identifier" => {
                    if type_name.is_none() {
                        type_name = self.get_node_text(&child, source);
                    }
                }
                "declaration_list" => {
                    // Extract methods in impl block
                    let mut method_cursor = child.walk();
                    for method_child in child.children(&mut method_cursor) {
                        if method_child.kind() == "function_item" {
                            if let Some(method_name) = self.find_child_text(&method_child, "identifier", source) {
                                let method_id = generate_node_id(&self.current_file, "method", &method_name);
                                methods.push(method_id);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Store methods for this type to link later
        if let Some(type_str) = type_name {
            if let Some(type_id) = self.symbol_to_id.get(&type_str) {
                self.type_methods
                    .entry(type_id.clone())
                    .or_insert_with(Vec::new)
                    .extend(methods);
            }
        }
    }

    fn extract_rust_macro(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let is_public = true; // macro_rules! macros are typically public in their module

        // Get macro name
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" {
                name = self.get_node_text(&child, source).unwrap_or_default();
                break;
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "macro", &name);
            let location = self.get_location(node);
            let pattern = self.get_node_text(node, source);

            let macro_node = Node::new(
                node_id.clone(),
                NodeKind::Macro(MacroNode {
                    name: name.clone(),
                    is_public,
                    macro_type: MacroType::Declarative,
                    pattern,
                }),
                location,
            );

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(macro_node);
        }
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

    fn extract_rust_constant(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;
        let mut value_type = None;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    is_public = true;
                }
                "identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                "type" => {
                    value_type = self.get_node_text(&child, source);
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "const", &name);
            let location = self.get_location(node);

            let const_node = Node::new(
                node_id.clone(),
                NodeKind::Constant(ConstantNode {
                    name: name.clone(),
                    value_type,
                    is_public,
                }),
                location,
            );

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(const_node);
        }
    }

    fn extract_rust_module(&mut self, node: &TSNode, source: &[u8]) {
        let mut name = String::new();
        let mut is_public = false;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "visibility_modifier" => {
                    is_public = true;
                }
                "identifier" => {
                    if name.is_empty() {
                        name = self.get_node_text(&child, source).unwrap_or_default();
                    }
                }
                _ => {}
            }
        }

        if !name.is_empty() {
            let node_id = generate_node_id(&self.current_file, "module", &name);
            let location = self.get_location(node);

            let module_node = Node::new(
                node_id.clone(),
                NodeKind::Module(ModuleNode {
                    name: name.clone(),
                    path: format!("{}::{}", self.current_file, name),
                    is_public,
                    children: Vec::new(),
                }),
                location,
            );

            self.symbol_to_id.insert(name.clone(), node_id.clone());
            self.file_symbols
                .entry(self.current_file.clone())
                .or_insert_with(Vec::new)
                .push(node_id.clone());
            self.graph.add_node(module_node);
        }
    }

    fn extract_rust_parameters(&mut self, node: &TSNode, source: &[u8]) -> Vec<Parameter> {
        let mut parameters = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "parameter" {
                let mut param_name = String::new();
                let mut param_type = None;
                let mut is_mutable = false;

                let mut param_cursor = child.walk();
                for param_child in child.children(&mut param_cursor) {
                    match param_child.kind() {
                        "identifier" | "self" => {
                            param_name =
                                self.get_node_text(&param_child, source).unwrap_or_default();
                        }
                        "type" => {
                            param_type = self.get_node_text(&param_child, source);
                        }
                        "mutable_specifier" => {
                            is_mutable = true;
                        }
                        _ => {}
                    }
                }

                if !param_name.is_empty() {
                    parameters.push(Parameter {
                        name: param_name,
                        param_type,
                        is_mutable,
                    });
                }
            }
        }

        parameters
    }

    fn extract_rust_fields(&mut self, node: &TSNode, source: &[u8]) -> Vec<Field> {
        let mut fields = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "field_declaration" {
                let mut field_name = String::new();
                let mut field_type = None;
                let mut is_public = false;

                let mut field_cursor = child.walk();
                for field_child in child.children(&mut field_cursor) {
                    match field_child.kind() {
                        "visibility_modifier" => {
                            is_public = true;
                        }
                        "field_identifier" => {
                            field_name =
                                self.get_node_text(&field_child, source).unwrap_or_default();
                        }
                        "type" => {
                            field_type = self.get_node_text(&field_child, source);
                        }
                        _ => {}
                    }
                }

                if !field_name.is_empty() {
                    fields.push(Field {
                        name: field_name,
                        field_type,
                        is_public,
                    });
                }
            }
        }

        fields
    }

    // Placeholder implementations for other languages
    fn process_python_node(&mut self, node: &TSNode, source: &[u8]) {
        // Python-specific extraction
        // Similar pattern to Rust but for Python syntax
        self.generic_traverse(node, source, "python");
    }

    fn process_js_ts_node(&mut self, node: &TSNode, source: &[u8], language: &str) {
        // JavaScript/TypeScript extraction
        self.generic_traverse(node, source, language);
    }

    fn process_go_node(&mut self, node: &TSNode, source: &[u8]) {
        // Go-specific extraction
        self.generic_traverse(node, source, "go");
    }

    fn process_java_node(&mut self, node: &TSNode, source: &[u8]) {
        // Java-specific extraction
        self.generic_traverse(node, source, "java");
    }

    fn process_c_cpp_node(&mut self, node: &TSNode, source: &[u8]) {
        // C/C++ extraction
        self.generic_traverse(node, source, "c/cpp");
    }

    fn generic_traverse(&mut self, node: &TSNode, source: &[u8], _language: &str) {
        // Generic traversal for unsupported languages or as fallback
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.process_node(&child, source, _language);
        }
    }

    fn resolve_edges(&mut self) {
        // Resolve pending function calls
        for (caller_id, callee_name) in &self.pending_calls {
            if let Some(callee_id) = self.symbol_to_id.get(callee_name) {
                self.graph.add_edge(Edge {
                    source: caller_id.clone(),
                    target: callee_id.clone(),
                    kind: EdgeKind::Calls,
                });
            }
        }

        // Resolve pending type references
        for (user_id, type_name) in &self.pending_type_refs {
            if let Some(type_id) = self.symbol_to_id.get(type_name) {
                self.graph.add_edge(Edge {
                    source: user_id.clone(),
                    target: type_id.clone(),
                    kind: EdgeKind::TypeReference,
                });
            }
        }

        // Calculate fan-in/fan-out for all nodes
        let node_ids: Vec<NodeId> = self.graph.nodes.keys().cloned().collect();
        for node_id in node_ids {
            let fan_in = self.graph.calculate_fan_in(&node_id);
            let fan_out = self.graph.calculate_fan_out(&node_id);

            if let Some(node) = self.graph.nodes.get_mut(&node_id) {
                node.metadata.fan_in = fan_in;
                node.metadata.fan_out = fan_out;
            }
        }
    }

    fn populate_file_symbols(&mut self) {
        // Populate file nodes with their symbol lists
        for (file_path, symbol_ids) in &self.file_symbols {
            let file_id = generate_node_id(file_path, "file", "");
            if let Some(node) = self.graph.nodes.get_mut(&file_id) {
                if let NodeKind::File(ref mut file_node) = node.kind {
                    file_node.symbols = symbol_ids.clone();
                }
            }
        }
    }

    fn attach_methods_to_types(&mut self) {
        // Attach methods to their respective types
        for (type_id, method_ids) in &self.type_methods {
            if let Some(node) = self.graph.nodes.get_mut(type_id) {
                if let NodeKind::Type(ref mut type_node) = node.kind {
                    type_node.methods = method_ids.clone();
                }
            }
        }
    }

    fn get_node_text(&self, node: &TSNode, source: &[u8]) -> Option<String> {
        let text = &source[node.start_byte()..node.end_byte()];
        String::from_utf8(text.to_vec()).ok()
    }

    fn get_location(&self, node: &TSNode) -> Location {
        Location {
            file: self.current_file.clone(),
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            start_col: node.start_position().column,
            end_col: node.end_position().column,
        }
    }
}

/// Build a graph from parsed files
pub fn build_graph(parsed_files: &[ParsedFile]) -> DocpackGraph {
    let builder = GraphBuilder::new();
    builder.build(parsed_files)
}
