package bridge

// MixerFixture holds 5D CNN3-style test volumes for the mixer bedrock.
type MixerFixture struct {
	XTest [][][][][]float64
}

func LoadMixerFixture(fixtureVersion, fixturesDir string) (*MixerFixture, error) {
	fx, err := LoadCNN3Fixture(fixtureVersion, fixturesDir)
	if err != nil {
		return nil, err
	}
	return &MixerFixture{XTest: fx.XTest}, nil
}

func SliceMixerTestInputs(fx *MixerFixture) [][][][][]float64 {
	return SliceCNN3TestInputs(&CNN3Fixture{XTest: fx.XTest}, MixerVolumeC, MixerVolumeD, MixerVolumeH, MixerVolumeW)
}
