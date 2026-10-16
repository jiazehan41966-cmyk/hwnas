# SCTD Source Investigation

## Conclusion

The downloaded dataset is classic SCTD / SCTD 1.0 from the original author GitHub repository, not SCTD 3.0.

## Original Paper And Recent-Paper Linkage

- Original dataset paper: Self-Trained Target Detection of Radar and Sonar Images Using Automatic Deep Learning.
- Recent references checked by task: ESL-YOLO (JMSE 13(8):1477, 2025) and MFF-YOLOv7 (JMSE 12(12):2326, 2024).
- The recent-paper SCTD identity matches the classic description: 357 images, three target classes, Pascal/VOC-style detection annotations in the original package.

## Original Download Link

- Repository: https://github.com/MingqiangNing/SCTD
- Raw archive: https://raw.githubusercontent.com/MingqiangNing/SCTD/master/SCTD.zip
- Archive SHA256: `329801ace811aced3dc7c842cc0d695877c5aae0a75545f5d544be399b8a0d9a`
- Current status: official GitHub raw download succeeded via Windows BITS.

## Mirrors

- No third-party mirror was used.
- GitHub codeload master zip attempt failed and was discarded after ZIP truncation evidence.
- SCTD 3.0 official entry points were checked only for version contrast:
  - https://github.com/automlresearch/SCTD-3.0
  - https://arxiv.org/abs/2608.08106

## Classic SCTD Versus SCTD 3.0

Classic SCTD 1.0 and SCTD 3.0 are not treated as the same data version. Classic SCTD 1.0 has 357 images and three classes in Pascal VOC XML. SCTD 3.0 is a separate 2026 dataset/release line and is not substituted here.

## Recommendation

Use this official GitHub classic SCTD 1.0 package for the planned classic-SCTD comparison. Do not replace it with SCTD 3.0 unless the experiment is explicitly re-scoped.
