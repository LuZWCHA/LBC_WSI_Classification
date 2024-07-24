# ALL SUPPORT FORMATS

openslide_support_formats = [".svs", ".tif", ".vms", ".vmu", ".ndpi", ".scn" , ".mrxs", ".tiff", ".svslide", ".bif"]
ibl_support_formats = [".ibl"]
sdpc_support_formats = [".sdpc"]
kfb_support_formats = [".kfb"]

SLIDE_FORMATS = {
    "openslide": openslide_support_formats,
    "iblsdk": ibl_support_formats,
    "sdpcdecode": sdpc_support_formats,
    "jiangfeng": kfb_support_formats
}


def get_default_support_slide_formats():
    return get_all_slide_formats(SLIDE_FORMATS)

def get_all_slide_formats(sdk_formats_map):
    ret = set()
    sdk_set = set()
    for sdk, support_formats in sdk_formats_map.items():
        for sf in support_formats:
            ret.add(sf)
        sdk_set.add(sdk)
    return ret, sdk_set