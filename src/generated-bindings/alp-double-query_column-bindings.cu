
#include <stdexcept>

#include "kernel-bindings.cuh"
#include "../engine/kernels.cuh"
#include "../engine/multi-column-host-kernels.cuh"

namespace bindings{

template<> bool query_column<double,flsgpu::device::ALPColumn<double>>(const flsgpu::device::ALPColumn<double> column, const unsigned unpack_n_vectors, const unsigned unpack_n_values, const enums::Unpacker unpacker, const enums::Patcher patcher , const double magic_value, const uint32_t n_samples){


if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Dummy ) {return kernels::host::query_column<double, 1, 1, flsgpu::device::ALPDecompressor<double, 1, flsgpu::device::BitUnpackerStatefulBranchless<double, 1, 1,  flsgpu::device::ALPFunctor<double, 1> >, flsgpu::device::DummyALPExceptionPatcher<double, 1, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}
if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Stateless ) {return kernels::host::query_column<double, 1, 1, flsgpu::device::ALPDecompressor<double, 1, flsgpu::device::BitUnpackerStatefulBranchless<double, 1, 1,  flsgpu::device::ALPFunctor<double, 1> >, flsgpu::device::StatelessALPExceptionPatcher<double, 1, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}
if (unpack_n_vectors == 1 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Stateful ) {return kernels::host::query_column<double, 1, 1, flsgpu::device::ALPDecompressor<double, 1, flsgpu::device::BitUnpackerStatefulBranchless<double, 1, 1,  flsgpu::device::ALPFunctor<double, 1> >, flsgpu::device::StatefulALPExceptionPatcher<double, 1, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}



if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Dummy ) {return kernels::host::query_column<double, 4, 1, flsgpu::device::ALPDecompressor<double, 4, flsgpu::device::BitUnpackerStatefulBranchless<double, 4, 1,  flsgpu::device::ALPFunctor<double, 4> >, flsgpu::device::DummyALPExceptionPatcher<double, 4, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}
if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Stateless ) {return kernels::host::query_column<double, 4, 1, flsgpu::device::ALPDecompressor<double, 4, flsgpu::device::BitUnpackerStatefulBranchless<double, 4, 1,  flsgpu::device::ALPFunctor<double, 4> >, flsgpu::device::StatelessALPExceptionPatcher<double, 4, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}
if (unpack_n_vectors == 4 && unpack_n_values == 1 && unpacker == enums::Unpacker::StatefulBranchless && patcher == enums::Patcher::Stateful ) {return kernels::host::query_column<double, 4, 1, flsgpu::device::ALPDecompressor<double, 4, flsgpu::device::BitUnpackerStatefulBranchless<double, 4, 1,  flsgpu::device::ALPFunctor<double, 4> >, flsgpu::device::StatefulALPExceptionPatcher<double, 4, 1>, flsgpu::device::ALPColumn<double>>, flsgpu::device::ALPColumn<double> >(column , magic_value, n_samples);}throw std::invalid_argument("Could not find correct binding in query_column ALP<double>");}

}
