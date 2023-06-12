package loss

type Loss interface {
	Backward()
	BackwardRetained()
	Value() float64
}
