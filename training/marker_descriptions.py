"""
1. Explicit biomarker grounding
2. Realistic H&E morphology
3. Immune lineage specificity
4. Spatial microenvironment context
5. Cross-tissue generalization
"""

MARKER_DESCRIPTIONS: dict[str, str] = {

    "Hoechst": (
        "Hematoxylin and eosin stained histologic section associated with strong nuclear "
        "signal corresponding to Hoechst nuclear counterstain, showing densely distributed "
        "nuclei across epithelial, stromal, vascular, and immune tissue compartments, "
        "including small hyperchromatic lymphocyte nuclei, vesicular epithelial nuclei "
        "with visible nucleoli, elongated spindle-cell nuclei within fibrous stroma, and "
        "scattered endothelial and inflammatory cell nuclei throughout the tissue "
        "microenvironment."
    ),

    "CD31": (
        "Hematoxylin and eosin stained histologic section associated with high CD31 "
        "(Platelet Endothelial Cell Adhesion Molecule-1) expression, showing endothelial "
        "cell-lined vascular and lymphatic channels composed of flattened cells with "
        "elongated cigar-shaped nuclei forming delicate luminal structures containing "
        "erythrocytes or pale proteinaceous fluid, distributed within reactive stroma, "
        "granulation tissue, angiogenic regions, and perivascular microenvironments."
    ),

    "CD45": (
        "Hematoxylin and eosin stained histologic section associated with high CD45 "
        "(Leukocyte Common Antigen) expression, showing dense leukocyte-rich inflammatory "
        "infiltrates composed predominantly of small mature lymphocytes with dark round "
        "nuclei and scant cytoplasm admixed with macrophages and other mononuclear immune "
        "cells, distributed throughout inflamed stroma, mucosal immune tissue, "
        "perivascular regions, and epithelial-stromal interfaces."
    ),

    "CD68": (
        "Hematoxylin and eosin stained histologic section associated with high CD68 "
        "(Cluster of Differentiation 68) expression, showing macrophage-rich stromal and "
        "inflammatory infiltrates composed of large histiocytic cells with abundant pale "
        "eosinophilic to vacuolated cytoplasm and eccentric oval to reniform nuclei, "
        "located within necrotic regions, granulation tissue, inflamed connective tissue, "
        "and tumour-associated immune microenvironments."
    ),

    "CD4": (
        "Hematoxylin and eosin stained histologic section associated with high CD4 "
        "(Cluster of Differentiation 4) expression, showing helper T-lymphocyte-rich "
        "immune infiltrates composed of small mature lymphocytes with condensed "
        "hyperchromatic nuclei distributed throughout immune-rich stroma, tertiary "
        "lymphoid aggregates, mucosal immune tissue, and epithelial-stromal interfaces, "
        "morphologically overlapping with other small T-cell populations."
    ),

    "FOXP3": (
        "Hematoxylin and eosin stained histologic section associated with high FOXP3 "
        "(Forkhead Box P3) expression, showing regulatory T-lymphocyte-rich immune "
        "microenvironments composed of small mature lymphocytes with dense round nuclei "
        "and minimal visible cytoplasm scattered within inflamed stroma, tertiary "
        "lymphoid structures, and epithelial-stromal interfaces, morphologically "
        "indistinguishable from other small lymphocytic populations on routine histology."
    ),

    "CD8a": (
        "Hematoxylin and eosin stained histologic section associated with high CD8a "
        "(Cluster of Differentiation 8 Alpha) expression, showing cytotoxic "
        "T-lymphocyte-rich inflammatory infiltrates composed of small mature lymphocytes "
        "with dense round basophilic nuclei and scant pale cytoplasm infiltrating between "
        "epithelial nests, along epithelial-stromal boundaries, and throughout "
        "immune-active stromal microenvironments."
    ),

    "CD45RO": (
        "Hematoxylin and eosin stained histologic section associated with high CD45RO "
        "(Cluster of Differentiation 45RO) expression, showing memory T-lymphocyte-rich "
        "immune infiltrates composed of mature lymphocytes slightly larger than resting "
        "lymphocytes with moderately condensed chromatin and thin rims of pale cytoplasm, "
        "forming dense immune aggregates and tertiary lymphoid structures within "
        "chronically inflamed stromal tissue."
    ),

    "CD20": (
        "Hematoxylin and eosin stained histologic section associated with high CD20 "
        "(Cluster of Differentiation 20) expression, showing B-lymphocyte-rich lymphoid "
        "follicles and tertiary lymphoid structures with pale germinal centre-like "
        "regions composed of activated lymphoid cells surrounded by darker mantle zones "
        "of small mature lymphocytes within chronically inflamed tissue and immune-rich "
        "stromal microenvironments."
    ),

    "PD-L1": (
        "Hematoxylin and eosin stained histologic section associated with high PD-L1 "
        "(Programmed Death-Ligand 1) expression, showing immune-reactive epithelial and "
        "stromal microenvironments with large polygonal epithelial or macrophage-like "
        "cells exhibiting vesicular nuclei and pale cytoplasm positioned adjacent to "
        "dense T-lymphocyte-rich inflammatory infiltrates at epithelial-stromal and "
        "tumour-immune interfaces."
    ),

    "CD3e": (
        "Hematoxylin and eosin stained histologic section associated with high CD3e "
        "(Cluster of Differentiation 3 Epsilon) expression, showing dense "
        "T-lymphocyte-rich inflammatory infiltrates composed of numerous small mature "
        "lymphocytes with round hyperchromatic nuclei, condensed chromatin, and scant "
        "cytoplasm distributed diffusely through inflamed stroma and infiltrating "
        "between epithelial structures and cohesive cell nests."
    ),

    "CD163": (
        "Hematoxylin and eosin stained histologic section associated with high CD163 "
        "(Cluster of Differentiation 163) expression, showing anti-inflammatory "
        "macrophage-rich stromal infiltrates composed of large histiocytic cells with "
        "abundant pale foamy cytoplasm, eccentric round to reniform nuclei, and "
        "occasional cytoplasmic vacuolation situated within fibrotic stroma, "
        "wound-healing microenvironments, perivascular connective tissue, and "
        "immunosuppressive tissue regions."
    ),

    "E-Cadherin": (
        "Hematoxylin and eosin stained histologic section associated with high "
        "E-Cadherin (Epithelial Cadherin / CDH1) expression, showing cohesive epithelial "
        "cell populations with preserved cell-cell adhesion, distinct lateral cell "
        "borders, and organised glandular, tubular, or sheet-like architecture forming "
        "structurally intact epithelial layers and cohesive epithelial nests sharply "
        "demarcated from surrounding stromal tissue."
    ),

    "Ki-67": (
        "Hematoxylin and eosin stained histologic section associated with high Ki-67 "
        "(Marker of Cellular Proliferation) expression, showing highly proliferative "
        "cellular regions with crowded enlarged nuclei, coarse chromatin, prominent "
        "nucleoli, elevated nuclear-to-cytoplasmic ratios, and frequent mitotic figures, "
        "concentrated within hypercellular tumour areas, regenerative epithelium, "
        "germinal centres, and actively cycling tissue compartments."
    ),

    "Pan-CK": (
        "Hematoxylin and eosin stained histologic section associated with high Pan-CK "
        "(Pan-Cytokeratin) expression, showing epithelial-cell-rich tissue composed of "
        "cohesive nests, glands, and sheet-like structures formed by polygonal epithelial "
        "cells with moderate pale eosinophilic cytoplasm and round vesicular nuclei, "
        "sharply separated from adjacent fibrous stroma, inflammatory infiltrates, and "
        "non-epithelial tissue compartments."
    ),

    "SMA": (
        "Hematoxylin and eosin stained histologic section associated with high SMA "
        "(Smooth Muscle Actin / Alpha-SMA) expression, showing reactive stromal and "
        "smooth-muscle-rich tissue composed of elongated spindle-shaped myofibroblasts "
        "and smooth muscle cells with cigar-shaped nuclei arranged in parallel fascicles "
        "and sweeping bundles surrounding epithelial structures, blood vessels, fibrotic "
        "septa, and desmoplastic stromal microenvironments."
    ),
}