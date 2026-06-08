package bridge

import "github.com/openfluke/loom/poly"

func manifestActivation(s string) poly.ActivationType {
	switch stringsLower(s) {
	case "relu":
		return poly.ActivationReLU
	case "tanh":
		return poly.ActivationTanh
	case "sigmoid":
		return poly.ActivationSigmoid
	case "linear":
		return poly.ActivationLinear
	default:
		return poly.ActivationLinear
	}
}

func stringsLower(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}
