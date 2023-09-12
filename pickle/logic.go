package pickle

import (
	"container/list"
	"encoding/binary"
)

var handlers = map[byte]func(*Decoder) error{
	bEmptyDict:  readEmptyDict,
	bBinPut:     readBinPut,
	bMark:       readMark,
	bBinUnicode: readBinUnicode,
	bGlobal:     readGlobal,
	bBinInt:     readBinInt,
}

func readEmptyDict(d *Decoder) error {
	d.variableStack.PushBack(newDict())
	return nil
}

func readBinPut(d *Decoder) error {
	idx, err := d.r.ReadByte()
	if err != nil {
		return err
	}
	d.memo[int(idx)] = d.variableStack.Back().Value.(*variable)
	return nil
}

func clear(list *list.List) {
	for elem := list.Front(); elem != nil; elem = elem.Next() {
		list.Remove(elem)
	}
}

func readMark(d *Decoder) error {
	d.markStack.PushBack(d.variableStack)
	clear(&d.variableStack)
	return nil
}

func readBinUnicode(d *Decoder) error {
	buf := make([]byte, 4)
	_, err := d.r.Read(buf)
	if err != nil {
		return err
	}
	size := binary.LittleEndian.Uint32(buf)
	buf = make([]byte, size)
	_, err = d.r.Read(buf)
	if err != nil {
		return err
	}
	d.variableStack.PushBack(newString(string(buf)))
	return nil
}

func readGlobal(d *Decoder) error {
	module, err := d.r.ReadBytes('\n')
	if err != nil {
		return err
	}
	name, err := d.r.ReadBytes('\n')
	if err != nil {
		return err
	}
	d.variableStack.PushBack(newClass(string(module[:len(module)-1]), string(name[:len(name)-1])))
	return nil
}

func readBinInt(d *Decoder) error {
	buf := make([]byte, 4)
	_, err := d.r.Read(buf)
	if err != nil {
		return err
	}
	n := int(binary.LittleEndian.Uint32(buf))
	if buf[3]&0x80 != 0 {
		n = -(int(^n) + 1)
	}
	d.variableStack.PushBack(newInt(n))
	return nil
}
