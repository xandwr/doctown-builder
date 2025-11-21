// Pipeline module declarations
// Each phase is a separate module for clean separation

pub mod analyze; // Phase 4: Compute complexity & metrics
pub mod cluster; // Phase 5: Semantic clustering
pub mod extract; // Phase 3: Extract definitions & relationships
pub mod generate; // Phase 6: LLM doc generation
pub mod ingest; // Phase 1: Extract files from zip/git
pub mod package; // Phase 7: Package outputs into .docpack
pub mod parse; // Phase 2: Generate ASTs via tree-sitter
