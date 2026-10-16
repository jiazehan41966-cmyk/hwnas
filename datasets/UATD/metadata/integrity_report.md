# UATD Integrity Report

- Generated at: 2026-09-17T07:24:14+00:00
- Raw file checksums: 4 archives downloaded and verified with Figshare MD5.
- Image count: 9200
- Annotation file count: 9203
- Parsed box count: 14639
- Classes: ['ball', 'circle cage', 'cube', 'cylinder', 'human body', 'metal bucket', 'plane', 'rov', 'square cage', 'tyre']
- Corrupted images: 0
- Empty annotation images: 0
- Duplicate filenames: 800
- Official split: {'provided': True, 'detected_from': 'Figshare archive names and extracted path segments', 'splits': {'test1': 800, 'test2': 800, 'train': 7600}, 'validation_split_detected': False, 'note': 'UATD provides training and two test archives; no fixed validation split was detected.'}
- Leakage note: Official train/test archives exist, but no validation split was detected. Do not create a random validation split until sequence/acquisition grouping is reviewed.

## Archive Files

- `UATD_Training.zip`: bytes=3869955319, md5=891b146871edcda313cdbd8cd6252f39, sha256=49a8c38e3cb98b04405b378c034a5e259249cd14102a3c8f3c1323e806cf5f7d
- `UATD_Test_1.zip`: bytes=421934004, md5=267818131310873c05950879bfec3e6f, sha256=9dce42baba978aba59d3601dbc086e744afa24c7565c628086eb35acf95f9b64
- `UATD_Test_2.zip`: bytes=424608381, md5=979479864b52925edca49074a09c1b3d, sha256=fbe8fd7f33063d735f51711fea46f7c057ffdea8c9b7b1cc74c6a8dcd6dc727e
- `UATD_OpenSLT.zip`: bytes=79823319, md5=d62542fa284a4cc1119687bfa9141754, sha256=da3f9b9f39231b08ffba7748264d5dc0ad55722926870f99d742999754daefb9

## Annotation Summary

- Annotation extensions: {'.json': 3, '.xml': 9200}
- Annotation formats: {'pascal_voc_xml': 14639}
- BBox formats: {'xyxy_pixel': 14639}
- Class counts: {'ball': 3463, 'circle cage': 860, 'cube': 2987, 'cylinder': 657, 'human body': 1434, 'metal bucket': 487, 'plane': 1065, 'rov': 1000, 'square cage': 1318, 'tyre': 1368}

## Image Summary

- Extensions: {'.bmp': 9200}
- Channel counts: {'3': 9200}
- Image modes: {'RGB': 9200}
- Resolution distribution: [{'width': 1024, 'height': 1026, 'count': 735}, {'width': 1024, 'height': 1028, 'count': 548}, {'width': 1024, 'height': 1257, 'count': 526}, {'width': 512, 'height': 1840, 'count': 452}, {'width': 512, 'height': 1227, 'count': 402}, {'width': 1024, 'height': 1542, 'count': 387}, {'width': 1024, 'height': 1541, 'count': 363}, {'width': 512, 'height': 1776, 'count': 359}, {'width': 1024, 'height': 1973, 'count': 353}, {'width': 1024, 'height': 1387, 'count': 260}, {'width': 1024, 'height': 1027, 'count': 235}, {'width': 512, 'height': 1841, 'count': 205}, {'width': 512, 'height': 1229, 'count': 203}, {'width': 1024, 'height': 1438, 'count': 180}, {'width': 1024, 'height': 1336, 'count': 166}, {'width': 1024, 'height': 1508, 'count': 160}, {'width': 512, 'height': 1168, 'count': 150}, {'width': 1024, 'height': 1540, 'count': 140}, {'width': 1024, 'height': 1735, 'count': 138}, {'width': 512, 'height': 1230, 'count': 118}, {'width': 512, 'height': 1844, 'count': 114}, {'width': 512, 'height': 1228, 'count': 113}, {'width': 512, 'height': 1167, 'count': 90}, {'width': 1024, 'height': 2041, 'count': 86}, {'width': 1024, 'height': 1961, 'count': 85}]

## Conversion Summary

```json
{
  "processed_yolo_root": "F:\\hwnas_datasets\\UATD\\processed_yolo",
  "converted_images": 9200,
  "converted_boxes": 14623,
  "combined_test_images": 1600,
  "combined_test_boxes": 2327,
  "skipped_boxes": 21,
  "split_counts": {
    "test": 1600,
    "test1": 800,
    "test2": 800,
    "train": 7600
  },
  "image_materialization": {
    "copy": 10800
  },
  "current_loader_compatibility": {
    "YoloDetectionDataset": true,
    "create_yolo_detection_dataloaders_train_valid_test": false,
    "reason": "No official validation split was detected; the project factory expects train/valid/test."
  }
}
```
