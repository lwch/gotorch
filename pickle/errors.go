package pickle

import "errors"

var ErrUnsupportedProtocol = errors.New("unsupported protocol")
var ErrInvalidOpcode = errors.New("invalid opcode")
