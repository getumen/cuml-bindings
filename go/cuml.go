package cuml4go

import "runtime"

func init() {
	runtime.LockOSThread()
}
