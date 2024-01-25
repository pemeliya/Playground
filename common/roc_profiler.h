
#include "common/common.h"
#include <rocprofiler/v2/rocprofiler.h>

#define CHECK_ROCPROFILER(call)                                     \
  if (auto res = (call); res != ROCPROFILER_STATUS_SUCCESS) {       \
      ThrowError< 256 >("ROCProfiler API error: %d: %s", __LINE__, rocprofiler_error_str(res));      \
  } 

class RocProfilerSession {

public:
  RocProfilerSession();

  ~RocProfilerSession();

  void start();
  void stop();

private:
  rocprofiler_session_id_t session_id;
  rocprofiler_buffer_id_t buffer_id;
};
