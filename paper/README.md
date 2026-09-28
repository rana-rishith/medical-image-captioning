# Accompanying Paper

**Title:** *Resource-Efficient Medical Image Captioning with a Frozen ViT-Base Encoder and Phi-2 under LoRA Fine-Tuning*
**Authors:** Rana Rishith Musunuri, Nandani Sharma (corresponding author)
**Affiliation:** Department of Information Technology, Manipal University Jaipur, India

The paper covers:

- Training a ViT-Base + Phi-2 captioning system on a single 12 GB RTX 3060: offline feature caching, LoRA, gradient checkpointing and the other measures that keep peak VRAM at 8.56 GiB
- The two-stage schedule and why joint training collapsed
- Full-test-split results on ROCOv2 (9,927 images) with a blind ablation and grounding probes
- An indicative comparison with DS@BioMed (ImageCLEFmedical 2024)
- Limitations

> The paper is available on request. Open an issue or contact the authors.
