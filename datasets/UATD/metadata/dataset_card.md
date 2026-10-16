# UATD Dataset Card

- Official name: Underwater acoustic target detection (UATD) dataset
- Official paper: https://www.nature.com/articles/s41597-022-01854-w (https://doi.org/10.1038/s41597-022-01854-w)
- Download URL: https://figshare.com/articles/dataset/UATD_Dataset/21331143; API https://api.figshare.com/v2/articles/21331143
- Sonar modality: multibeam forward-looking sonar (MFLS), from the official paper metadata.
- Sonar device: Tritech Gemini 1200ik, from the official paper metadata.
- Number of images: 9200
- Classes: ball, circle cage, cube, cylinder, human body, metal bucket, plane, rov, square cage, tyre
- Class count: 10
- Annotation type: {'pascal_voc_xml': 14639}
- BBox format: {'xyxy_pixel': 14639}
- Image extensions: {'.bmp': 9200}
- Image resolutions: 1024x1026 (735), 1024x1028 (548), 1024x1257 (526), 512x1840 (452), 512x1227 (402), 1024x1542 (387), 1024x1541 (363), 512x1776 (359), 1024x1973 (353), 1024x1387 (260), 1024x1027 (235), 512x1841 (205)
- Image channels/modes: channels={'3': 9200}, modes={'RGB': 9200}
- Official split information: {'provided': True, 'detected_from': 'Figshare archive names and extracted path segments', 'splits': {'test1': 800, 'test2': 800, 'train': 7600}, 'validation_split_detected': False, 'note': 'UATD provides training and two test archives; no fixed validation split was detected.'}
- Detected leakage risks: Official train/test archives exist, but no validation split was detected. Do not create a random validation split until sequence/acquisition grouping is reviewed.
- Current project-compatible format: YOLO under datasets/UATD/processed_yolo
- Conversion procedure: `python scripts/prepare_uatd.py --root datasets/UATD`

## Downloaded Files

- `UATD_Training.zip`: 3869955319 bytes, md5 `891b146871edcda313cdbd8cd6252f39`, sha256 `49a8c38e3cb98b04405b378c034a5e259249cd14102a3c8f3c1323e806cf5f7d`
- `UATD_Test_1.zip`: 421934004 bytes, md5 `267818131310873c05950879bfec3e6f`, sha256 `9dce42baba978aba59d3601dbc086e744afa24c7565c628086eb35acf95f9b64`
- `UATD_Test_2.zip`: 424608381 bytes, md5 `979479864b52925edca49074a09c1b3d`, sha256 `fbe8fd7f33063d735f51711fea46f7c057ffdea8c9b7b1cc74c6a8dcd6dc727e`
- `UATD_OpenSLT.zip`: 79823319 bytes, md5 `d62542fa284a4cc1119687bfa9141754`, sha256 `da3f9b9f39231b08ffba7748264d5dc0ad55722926870f99d742999754daefb9`

## Notes

The raw archives and extracted source tree are kept under `datasets/UATD/raw/`.
No random validation split was created. The official source provides training
and two test archives; validation design needs a separate group/leakage review
before fixed-architecture transfer experiments.
