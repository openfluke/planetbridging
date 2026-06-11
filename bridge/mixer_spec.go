package bridge

// MixerAllV1 is the fixed layer stack for mixer_all_v1 (all 7 Loom layer types).
//
// Pipeline (tensor ranks shown):
//   CNN3 [B,1,2,2,2] → flat 2
//   Dense 2→8 → reshape [B,1,2,4]
//   CNN2 → flat 6
//   Dense 6→8 relu → reshape [B,1,8]
//   CNN1 → flat 4
//   Dense 4→8 → reshape [B,2,4]
//   MHA → RNN → LSTM → flat 8
//   Dense head 8→8
const (
	MixerModelID       = "mixer_all_v1"
	MixerModelIDV2     = "mixer_all_v2"
	MixerFixtureVer    = "mixer_bedrock_v1"
	MixerLayerCount    = 10
	MixerLayerCountV2  = 16
	MixerEmbedVocab    = 16
	MixerEmbedDim      = 4
	MixerEmbedSeq      = 2
	MixerSwiGLUInter   = 8
	MixerOutputDim     = 8
	MixerVolumeC       = 1
	MixerVolumeD       = 2
	MixerVolumeH       = 2
	MixerVolumeW       = 2
	MixerCNN3Filters   = 2
	MixerCNN3Kernel    = 2
	MixerDenseBridge1  = 8  // after cnn3 flat 2
	MixerCNN2H         = 2
	MixerCNN2W         = 4
	MixerCNN2Filters   = 2
	MixerCNN2Kernel    = 2
	MixerDenseBridge2  = 8  // after cnn2 flat 6
	MixerCNN1Len       = 8
	MixerCNN1Filters   = 4
	MixerCNN1Kernel    = 8
	MixerDenseBridge3  = 8  // after cnn1 flat 4
	MixerMHASeq        = 2
	MixerMHADModel     = 4
	MixerMHAHeads      = 2
	MixerMHAHeadDim    = 2
	MixerRecurrentSeq  = 2
	MixerRecurrentIn   = 4
	MixerRecurrentHid  = 4
)

// MixerLayerCountForModel returns expected layer count for a mixer model id.
func MixerLayerCountForModel(modelID string) int {
	if modelID == MixerModelIDV2 {
		return MixerLayerCountV2
	}
	return MixerLayerCount
}

// IsMixerV2 reports whether modelID is the full 12-type stack.
func IsMixerV2(modelID string) bool {
	return modelID == MixerModelIDV2
}

// MixerPipelineDescription returns a one-line layer chain for PDF / compare logs.
func MixerPipelineDescription(modelID string) string {
	switch modelID {
	case MixerModelIDV2:
		return "CNN3→Dense→CNN2→Dense→CNN1→Dense→Embedding→LayerNorm→MHA→Residual→RMSNorm→SwiGLU→Residual→RNN→LSTM→Dense (16 layers, all 12 types)"
	case MixerModelID:
		return "CNN3→Dense→CNN2→Dense→CNN1→Dense→MHA→RNN→LSTM→Dense (10 layers, 7 types)"
	default:
		return ""
	}
}
