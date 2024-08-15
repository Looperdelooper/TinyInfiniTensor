#include "utils/operator_utils.h"
#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
    if (inputs.size() != 2) {
        return {};
    }

    const auto& dimsA = inputs[0]->getDims();
    const auto& dimsB = inputs[1]->getDims();
    auto outputDims = infer_broadcast(dimsA, dimsB);
    size_t lastDimIndex = outputDims.size() - 1;
    if (this->transA) {
        outputDims[lastDimIndex - 1] = dimsA[lastDimIndex];
    } else {
        outputDims[lastDimIndex - 1] = dimsA[lastDimIndex - 1];
    }

    if (this->transB) {
        outputDims[lastDimIndex] = dimsB[lastDimIndex - 1];
    } else {
        outputDims[lastDimIndex] = dimsB[lastDimIndex];
    }

    return {{outputDims}};
    }

} // namespace infini