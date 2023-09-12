package pickle

import (
	"bufio"
	"container/list"
	"fmt"
	"io"
)

type Decoder struct {
	version       int
	r             *bufio.Reader
	markStack     list.List
	variableStack list.List
	memo          map[int]*variable
}

func NewDecoder(r io.Reader) (*Decoder, error) {
	rd := bufio.NewReader(r)
	protocol, err := rd.ReadByte()
	if err != nil {
		return nil, err
	}
	if protocol != 0x80 {
		return nil, ErrUnsupportedProtocol
	}
	version, err := rd.ReadByte()
	if err != nil {
		return nil, err
	}
	return &Decoder{
		version: int(version),
		r:       rd,
		memo:    make(map[int]*variable),
	}, nil
}

func (d *Decoder) Unmarshal(v interface{}) error {
	for {
		opcode, err := d.r.ReadByte()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		handler, ok := handlers[opcode]
		if !ok {
			fmt.Printf("unsupported opcode: 0x%x(%c)\n", opcode, opcode)
			return ErrInvalidOpcode
		}
		handler(d)
	}
}
