package host

// ActiveDensePlanets are the Python engines in scope for dense bedrock + compare UI.
// Paddle is intentionally excluded (maintenance / conda cost vs coverage).
var ActiveDensePlanets = map[string]bool{
	"pytorch":    true,
	"tensorflow": true,
	"jax":        true,
	"sklearn":    true,
}

func isActivePlanet(planet string) bool {
	return ActiveDensePlanets[planet]
}

func filterActiveReports(reports []Report) []Report {
	out := make([]Report, 0, len(reports))
	for _, r := range reports {
		r.Normalize()
		if isActivePlanet(r.Planet) {
			out = append(out, r)
		}
	}
	return out
}
