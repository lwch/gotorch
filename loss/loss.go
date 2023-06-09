package loss

type Loss interface {
	Backward()
	Value() float64
}
