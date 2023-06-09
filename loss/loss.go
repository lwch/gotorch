package loss

type Loss interface {
	Backward()
}
