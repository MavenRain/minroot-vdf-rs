//! The `MinRoot` pipeline as a free category.
//!
//! The fifth-root datapath has three stages:
//!
//! ```text
//!   PreSqr --[SQR]--> PostSqr --[MUL]--> PostMul --[RED]--> PostRed
//! ```
//!
//! Each stage is an **edge** in the pipeline graph.  Objects (vertices)
//! are the polynomial representations at each stage.  Morphisms (paths)
//! are compositions of stages.
//!
//! The full fifth-root computation is the path `RED . MUL . SQR` iterated
//! 258 times (one per exponent bit, plus overhead).
//!
//! By encoding the pipeline as a free category, we separate the topology
//! (what connects to what) from the implementation (how each stage computes).
//! The free category's universal property gives us [`interpret`]: any
//! graph morphism into a target category (e.g., RHDL circuits) extends
//! uniquely to a functor, producing the composed hardware pipeline.
//!
//! [`interpret`]: comp_cat_rs::collapse::free_category::interpret

use comp_cat_rs::collapse::free_category::{
    Edge, FreeCategoryError, Graph, Vertex,
};

/// Vertex indices in the pipeline graph.
///
/// These correspond to the data representation at each stage
/// of the fifth-root computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineVertex {
    /// Input to the squaring stage.
    PreSquare,
    /// Output of squaring, input to multiplication.
    PostSquare,
    /// Output of multiplication, input to reduction.
    PostMultiply,
    /// Output of reduction (feeds back to `PreSquare` in the ring).
    PostReduce,
}

impl PipelineVertex {
    /// All vertices in pipeline order.
    pub const ALL: [Self; 4] = [
        Self::PreSquare,
        Self::PostSquare,
        Self::PostMultiply,
        Self::PostReduce,
    ];

    /// Convert to a free-category [`Vertex`].
    #[must_use]
    pub fn to_vertex(self) -> Vertex {
        Vertex::new(self as usize)
    }
}

/// Edge indices in the pipeline graph.
///
/// These correspond to the pipeline stage computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineEdge {
    /// Polynomial squaring with reduction: `PreSquare -> PostSquare`.
    SquareReduce,
    /// Conditional multiplication by base: `PostSquare -> PostMultiply`.
    Multiply,
    /// Montgomery reduction: `PostMultiply -> PostReduce`.
    Reduce,
}

impl PipelineEdge {
    /// All edges in pipeline order.
    pub const ALL: [Self; 3] = [
        Self::SquareReduce,
        Self::Multiply,
        Self::Reduce,
    ];

    /// Convert to a free-category [`Edge`].
    #[must_use]
    pub fn to_edge(self) -> Edge {
        Edge::new(self as usize)
    }
}

/// The `MinRoot` fifth-root pipeline graph.
///
/// ```text
///   PreSquare --[SquareReduce]--> PostSquare --[Multiply]--> PostMultiply --[Reduce]--> PostReduce
/// ```
///
/// This graph generates the free category whose paths represent
/// composed pipeline computations.
#[derive(Debug, Clone, Copy)]
pub struct PipelineGraph;

impl Graph for PipelineGraph {
    fn vertex_count(&self) -> usize {
        PipelineVertex::ALL.len()
    }

    fn edge_count(&self) -> usize {
        PipelineEdge::ALL.len()
    }

    fn source(&self, edge: Edge) -> Result<Vertex, FreeCategoryError> {
        match edge.index() {
            0 => Ok(PipelineVertex::PreSquare.to_vertex()),
            1 => Ok(PipelineVertex::PostSquare.to_vertex()),
            2 => Ok(PipelineVertex::PostMultiply.to_vertex()),
            _ => Err(FreeCategoryError::EdgeOutOfBounds {
                edge,
                count: self.edge_count(),
            }),
        }
    }

    fn target(&self, edge: Edge) -> Result<Vertex, FreeCategoryError> {
        match edge.index() {
            0 => Ok(PipelineVertex::PostSquare.to_vertex()),
            1 => Ok(PipelineVertex::PostMultiply.to_vertex()),
            2 => Ok(PipelineVertex::PostReduce.to_vertex()),
            _ => Err(FreeCategoryError::EdgeOutOfBounds {
                edge,
                count: self.edge_count(),
            }),
        }
    }
}

/// Constructs the full single-round path: `SQR ; MUL ; RED`.
///
/// This path goes from `PreSquare` to `PostReduce` through all three stages.
///
/// # Errors
///
/// Returns [`FreeCategoryError`] if path construction fails (should not
/// happen for the well-formed pipeline graph).
pub fn single_round_path() -> Result<comp_cat_rs::collapse::free_category::Path, FreeCategoryError>
{
    let graph = PipelineGraph;
    let sqr =
        comp_cat_rs::collapse::free_category::Path::singleton(&graph, PipelineEdge::SquareReduce.to_edge())?;
    let mul =
        comp_cat_rs::collapse::free_category::Path::singleton(&graph, PipelineEdge::Multiply.to_edge())?;
    let red =
        comp_cat_rs::collapse::free_category::Path::singleton(&graph, PipelineEdge::Reduce.to_edge())?;
    sqr.compose(mul).and_then(|p| p.compose(red))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_graph_has_4_vertices_3_edges() {
        let graph = PipelineGraph;
        assert_eq!(graph.vertex_count(), 4);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn single_round_path_connects_pre_to_post() {
        let path = single_round_path();
        assert!(path.iter().all(|p| {
            p.source() == PipelineVertex::PreSquare.to_vertex()
                && p.target() == PipelineVertex::PostReduce.to_vertex()
                && p.len() == 3
        }));
    }

    #[test]
    fn edges_are_composable() {
        let graph = PipelineGraph;
        // SQR target == MUL source
        let sqr_target = graph.target(PipelineEdge::SquareReduce.to_edge());
        let mul_source = graph.source(PipelineEdge::Multiply.to_edge());
        assert_eq!(sqr_target.ok(), mul_source.ok());

        // MUL target == RED source
        let mul_target = graph.target(PipelineEdge::Multiply.to_edge());
        let red_source = graph.source(PipelineEdge::Reduce.to_edge());
        assert_eq!(mul_target.ok(), red_source.ok());
    }
}
