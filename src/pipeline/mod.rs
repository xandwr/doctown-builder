// Pipeline module declarations
// Each phase is a separate module for clean separation

pub mod ingest;    // Phase 1: Extract files from zip/git
pub mod parse;     // Phase 2: Generate ASTs via tree-sitter
pub mod extract;   // Phase 3: Extract definitions & relationships
pub mod analyze;   // Phase 4: Compute complexity & metrics
pub mod cluster;   // Phase 5: Semantic clustering
pub mod generate;  // Phase 6: LLM doc generation
