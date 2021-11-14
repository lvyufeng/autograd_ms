from mindspore.common import tensor
import mindspore.ops.operations as P

def _tensor_getitem(data, index):
    # if isinstance(index, 'Tensor'):
    #     return tensor_index_by_tensor(data, index)
    if isinstance(index, list):
        pass
    if isinstance(index, tuple):
        pass
    if isinstance(index, bool):
        pass
    if isinstance(index, int):
        pass
    if isinstance(index, slice):
        return tensor_index_by_slice(data, index)
    if index is None:
        return P.ExpandDims()(data, 0)
    if index is ...:
        return data
    raise IndexError(f"Only support integers, slices(`:`), ellipsis(`...`), None, bool, tensor with int, "
                     f"list and tuple ,but got {index} with type {type(index)}.")

def _tensor_setitem():
    pass

def tensor_index_by_tensor(data, tensor_index):
    return P.Gather()(data, tensor_index, 0)

def tensor_index_by_slice(data, slice_index):
    """Tensor getitem by a slice."""
    data_shape = data.shape
    is_dynamic = (-1 in data_shape)
    if is_dynamic:
        return tensor_index_by_dyn_slice(data, slice_index)
    begin_strides, end_strides, step_strides = get_stride_info_from_slice(data_shape, slice_index)
    return P.StridedSlice()(data, begin_strides, end_strides, step_strides)

def tensor_index_by_dyn_slice(data, slice_index):
    """Tensor getitem by a slice."""
    data_dims = data.ndim
    data_shape = P.DynamicShape()(data)
    begin_strides, end_strides, step_strides = [], [], []
    start, stop, step = get_slice_stride(slice_index, data_shape[0])
    begin_strides.append(start)
    end_strides.append(stop)
    step_strides.append(step)

    for index in range(1, data_dims):
        begin_strides.append(0)
        end_strides.append(data_shape[index])
        step_strides.append(1)
    begin_tensor = tuple(begin_strides)
    end_tensor = tuple(end_strides)
    step_tensor = tuple(step_strides)
    return P.StridedSlice()(data, begin_tensor, end_tensor, step_tensor)

def get_stride_info_from_slice(data_shape, slice_index):
    """Get stride info from a python slice"""
    begin, end, step = get_slice_stride(slice_index, data_shape[0])
    begin_strides = [begin]
    end_strides = [end]
    step_strides = [step]
    for end in data_shape[1:]:
        begin_strides.append(0)
        end_strides.append(end)
        step_strides.append(1)
    return tuple(begin_strides), tuple(end_strides), tuple(step_strides)

def get_slice_stride(index_slice, dim_size):
    """Get slice stride info"""
    step = 1 if index_slice.step is None else index_slice.step
    start_default = 0
    stop_default = dim_size
    if step < 0:
        start_default = -1
        stop_default = -(dim_size + 1)
    start = start_default if index_slice.start is None else index_slice.start
    stop = stop_default if index_slice.stop is None else index_slice.stop
    return start, stop, step