import sys
import pydicom

# Tags we care about for preprocessing + consistency checks
TAGS = [
    ("Modality", (0x0008, 0x0060)),
    ("SOPClassUID", (0x0008, 0x0016)),
    ("Manufacturer", (0x0008, 0x0070)),
    ("ManufacturerModelName", (0x0008, 0x1090)),
    ("SoftwareVersions", (0x0018, 0x1020)),
    ("DeviceSerialNumber", (0x0018, 0x1000)),  # you'll redact
    ("TransducerData", (0x0018, 0x5010)),       # often present in US
    ("ImageType", (0x0008, 0x0008)),
    ("PhotometricInterpretation", (0x0028, 0x0004)),
    ("SamplesPerPixel", (0x0028, 0x0002)),
    ("Rows", (0x0028, 0x0010)),
    ("Columns", (0x0028, 0x0011)),
    ("BitsAllocated", (0x0028, 0x0100)),
    ("BitsStored", (0x0028, 0x0101)),
    ("HighBit", (0x0028, 0x0102)),
    ("PixelRepresentation", (0x0028, 0x0103)),
    ("RescaleIntercept", (0x0028, 0x1052)),
    ("RescaleSlope", (0x0028, 0x1053)),
    ("WindowCenter", (0x0028, 0x1050)),
    ("WindowWidth", (0x0028, 0x1051)),
    ("PixelSpacing", (0x0028, 0x0030)),
    ("FrameTime", (0x0018, 0x1063)),
    ("NumberOfFrames", (0x0028, 0x0008)),
    ("BurnedInAnnotation", (0x0028, 0x0301)),
    ("LossyImageCompression", (0x0028, 0x2110)),
    ("LossyImageCompressionMethod", (0x0028, 0x2114)),
]

def get(ds, name, tag):
    return ds.get(tag, "MISSING")

def main(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    print(f"FILE: {path}\n---")
    for name, tag in TAGS:
        val = get(ds, name, tag)
        # keep it single-line
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            val = list(val)
        print(f"{name}: {val}")
    print("---\nNOTE: Redact PatientName/PatientID/Accession/StudyInstanceUID if present elsewhere.\n")

if __name__ == "__main__":
    main(sys.argv[1])
