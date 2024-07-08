package optimizer

type Optimizer interface {
	Step()
	GetLr() float64
	SetLr(float64)
}
