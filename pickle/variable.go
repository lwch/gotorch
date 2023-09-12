package pickle

type variableType int

const (
	tClass variableType = iota
	tDict
	tList
	tString
	tInt
)

type variable struct {
	t    variableType
	data interface{}
}

func newDict() *variable {
	return &variable{
		t:    tDict,
		data: make(map[string]*variable),
	}
}

func newString(str string) *variable {
	return &variable{
		t:    tString,
		data: str,
	}
}

type class struct {
	module string
	name   string
}

func newClass(module, name string) *variable {
	return &variable{
		t:    tClass,
		data: &class{module, name},
	}
}

func newInt(n int) *variable {
	return &variable{
		t:    tInt,
		data: n,
	}
}
