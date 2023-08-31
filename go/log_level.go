package cuml4go

type LogLevel int

const (
	Off      LogLevel = 0
	Critical LogLevel = 1
	Error    LogLevel = 2
	Warn     LogLevel = 3
	Info     LogLevel = 4
	Debug    LogLevel = 5
	Trace    LogLevel = 6
)
