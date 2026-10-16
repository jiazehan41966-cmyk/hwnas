# Classic SCTD Dataset Card

## Identity

- Dataset: classic SCTD / SCTD 1.0
- Full name used by source repository: Sonar Common Target Detection Dataset (SCTD 1.0)
- Source repository: https://github.com/MingqiangNing/SCTD
- Source archive: https://raw.githubusercontent.com/MingqiangNing/SCTD/master/SCTD.zip
- SHA256: `329801ace811aced3dc7c842cc0d695877c5aae0a75545f5d544be399b8a0d9a`
- Original paper: Self-Trained Target Detection of Radar and Sonar Images Using Automatic Deep Learning
- Imaging mode: side-scan sonar according to recent paper descriptions of SCTD; not SAS-only SCTD 3.0
- Version: 1.0
- Release evidence: source repository history and README identify this package as SCTD 1.0; repository files are dated 2020-09-11 after extraction.

## Scope

This is not SCTD 3.0. SCTD 3.0 is treated only as a version contrast item and was not downloaded or used.

## Raw Data

- Images: 357
- Annotations: 357
- Objects: 363
- Classes: {'aircraft': 57, 'human': 35, 'ship': 271}
- Image resolutions: {'415x385': 1, '300x188': 1, '743x419': 1, '442x340': 1, '1109x596': 1, '355x196': 1, '221x194': 1, '272x210': 1, '234x382': 1, '188x197': 1, '220x117': 1, '322x202': 1, '459x197': 1, '404x254': 1, '439x229': 1, '501x300': 1, '345x250': 2, '1146x904': 1, '357x300': 1, '1400x1050': 1, '214x236': 1, '296x170': 1, '255x198': 1, '287x175': 1, '191x264': 1, '259x194': 3, '184x274': 1, '311x162': 1, '660x568': 1, '1251x1250': 1, '360x384': 1, '460x281': 1, '810x758': 1, '1364x1365': 1, '740x416': 1, '534x401': 1, '400x302': 1, '2550x1588': 1, '341x428': 1, '333x403': 1, '402x512': 1, '512x509': 1, '293x283': 1, '620x745': 1, '791x589': 1, '561x681': 1, '611x894': 1, '796x466': 1, '1135x524': 1, '817x787': 1, '809x780': 1, '926x694': 1, '673x836': 1, '373x515': 1, '201x162': 1, '965x725': 1, '460x351': 2, '330x187': 2, '207x215': 2, '317x410': 1, '512x1024': 2, '394x476': 1, '438x330': 1, '1600x768': 1, '224x225': 1, '218x231': 2, '266x190': 2, '766x753': 2, '807x900': 2, '980x673': 1, '1110x673': 2, '1232x745': 2, '1032x751': 1, '1699x638': 2, '185x158': 2, '222x227': 3, '1354x864': 1, '744x497': 1, '860x460': 1, '950x360': 1, '410x230': 2, '640x480': 1, '862x575': 1, '491x275': 1, '414x390': 1, '850x956': 1, '220x248': 1, '940x505': 1, '447x497': 1, '929x600': 1, '699x457': 1, '894x472': 1, '414x242': 1, '616x530': 1, '903x486': 1, '972x497': 1, '539x544': 1, '975x600': 1, '465x289': 1, '956x375': 1, '399x271': 1, '270x303': 1, '445x179': 1, '1197x585': 1, '595x633': 1, '814x683': 1, '940x627': 1, '800x480': 1, '550x459': 1, '800x431': 1, '371x550': 1, '563x550': 1, '800x382': 1, '473x373': 1, '702x468': 1, '790x388': 1, '1440x550': 1, '520x324': 1, '1280x427': 1, '724x470': 1, '600x323': 1, '600x412': 1, '250x252': 1, '804x445': 1, '940x705': 1, '981x1045': 1, '400x300': 1, '436x654': 1, '375x292': 1, '725x520': 1, '320x230': 1, '430x430': 1, '640x568': 1, '449x593': 2, '285x177': 1, '300x168': 3, '266x189': 2, '286x176': 1, '279x180': 1, '423x119': 1, '263x191': 1, '236x214': 1, '269x187': 1, '252x176': 1, '174x290': 2, '237x212': 1, '268x188': 2, '265x190': 1, '271x186': 1, '274x184': 2, '303x167': 1, '218x232': 3, '255x197': 1, '281x179': 2, '225x225': 1, '215x150': 1, '220x214': 1, '321x157': 1, '216x234': 1, '275x183': 2, '212x238': 3, '295x171': 1, '318x159': 1, '178x283': 1, '234x216': 2, '187x199': 2, '288x175': 1, '302x167': 1, '292x172': 1, '383x132': 1, '273x184': 1, '320x157': 1, '254x198': 1, '335x150': 1, '249x202': 1, '304x166': 1, '204x247': 1, '352x368': 1, '509x455': 1, '600x450': 1, '1280x720': 1, '600x800': 1, '1200x912': 1, '850x464': 1, '2000x1731': 1, '545x539': 1, '1024x576': 2, '1001x631': 1, '701x345': 1, '860x485': 1, '800x800': 1, '800x732': 1, '528x274': 1, '434x526': 1, '435x1600': 1, '352x450': 1, '800x852': 1, '358x224': 1, '356x364': 1, '622x1026': 1, '870x290': 1, '285x204': 1, '355x500': 1, '800x592': 1, '512x280': 3, '434x299': 1, '512x358': 4, '402x250': 1, '512x249': 2, '512x244': 3, '371x512': 2, '512x310': 3, '310x440': 1, '512x162': 1, '512x441': 1, '582x676': 1, '444x289': 1, '317x211': 1, '374x135': 1, '697x597': 1, '665x762': 1, '479x828': 1, '1010x633': 1, '803x810': 1, '687x743': 1, '1453x740': 1, '738x806': 1, '1017x521': 1, '522x557': 1, '528x538': 1, '457x551': 1, '515x625': 1, '2000x795': 1, '613x581': 1, '617x544': 1, '781x756': 1, '586x772': 1, '715x629': 1, '659x830': 1, '866x742': 1, '188x154': 1, '1105x632': 1, '1176x648': 1, '812x539': 1, '965x682': 1, '820x539': 1, '826x596': 1, '408x394': 1, '525x653': 1, '435x536': 1, '1136x473': 1, '766x606': 1, '545x746': 1, '776x766': 1, '1080x533': 1, '568x409': 1, '408x588': 1, '596x849': 1, '351x597': 1, '579x715': 1, '1098x668': 1, '463x459': 1, '699x277': 1, '1077x540': 1, '452x430': 1, '527x582': 1, '966x570': 1, '696x636': 1, '449x748': 1, '456x709': 1, '867x481': 1, '479x545': 1, '509x586': 1, '729x746': 1, '395x416': 1, '468x606': 1, '728x865': 1, '607x621': 1, '632x724': 1, '659x691': 1, '747x684': 1, '502x766': 1, '356x713': 1, '522x862': 1, '405x428': 1, '658x359': 1, '690x681': 1, '620x888': 1, '603x704': 1, '426x711': 1, '714x619': 1, '659x493': 1, '608x883': 1, '703x798': 1, '881x841': 1, '1123x425': 1, '1369x741': 1, '668x687': 1, '923x499': 1, '840x404': 1, '505x617': 1, '740x361': 1, '451x739': 1, '534x829': 1, '599x593': 1, '708x818': 1, '388x565': 1, '481x574': 1, '345x474': 1, '393x409': 1, '660x811': 1, '591x576': 1}
- Raw annotation format: Pascal VOC XML with pixel-space `xmin ymin xmax ymax`.
- Official split files found: {'ImageSets/Main/trainval.txt': {'count': 357, 'first_ids': ['000002', '000006', '000007', '000008', '000010'], 'last_ids': ['000129', '000130', '000131', '000133', '000134']}}

## Processed View

- Root: `datasets\SCTD\processed`
- Annotation format: YOLO normalized `class cx cy w h`.
- Class mapping: `datasets\SCTD\metadata\class_mapping.json`
- Split policy: project_derived_numeric_run_group_split_v1; this is project-derived, not an official SCTD split.
- Split counts: {'train': 249, 'valid': 54, 'test': 54}
- Image materialization: {'hardlink': 357}

## Leakage Risk

The source package has numeric filenames with many consecutive runs: 46 runs of length >= 3 were observed. Acquisition metadata is absent, so consecutive-frame leakage cannot be certified away. The processed split keeps consecutive numeric runs together to reduce obvious leakage risk, but this is not a substitute for mission-level metadata.
