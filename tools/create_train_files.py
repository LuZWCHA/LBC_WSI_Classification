import os

def create_shishi_train_files():
    """by huanggf"""
    from datasets.our_datasets import SHISHI_RAW
    
    dirpath_wsi_path = SHISHI_RAW.root_path
    
    dirnames_train = SHISHI_RAW.get_default_train_datasets()
    dirnames_test = SHISHI_RAW.get_default_test_datasets()
    filename_anno = SHISHI_RAW.default_anno_file_name
    
    filepath_anno_wsi_path_train = SHISHI_RAW.at(SHISHI_RAW.default_train_file_path)
    filepath_anno_wsi_path_test = SHISHI_RAW.at(SHISHI_RAW.default_test_file_path)
    
    lines_total = []
    for dirname_sub in dirnames_train:
        dirpath_sub = '%s/%s' % (dirpath_wsi_path, dirname_sub)
        filepath_anno = os.path.join(dirpath_sub, filename_anno)
        
        f = open(filepath_anno)
        lines = f.readlines()
        f.close()
        print(dirname_sub, len(lines))
        
        for line in lines:
            arrs = line.split(' ')
            filename_img = arrs[0]
            
            filepath_img = '%s/images/%s' % (dirpath_sub, filename_img)
            line_new = line.replace(filename_img, filepath_img)
            if not line_new.endswith('\n'):
                line_new = '%s\n' % (line_new)
            lines_total.append(line_new)
    print('\ntotal train: ', len(lines_total))
    
    f = open(filepath_anno_wsi_path_train, 'w')
    f.writelines(lines_total)
    f.close()
    
    lines_total = []
    for dirname_sub in dirnames_test:
        dirpath_sub = '%s/%s' % (dirpath_wsi_path, dirname_sub)
        filepath_anno = os.path.join(dirpath_sub, filename_anno)
        
        f = open(filepath_anno)
        lines = f.readlines()
        f.close()
        print(dirname_sub, len(lines))
        
        for line in lines:
            arrs = line.split(' ')
            filename_img = arrs[0]
            filepath_img = '%s/images/%s' % (dirpath_sub, filename_img)
            line_new = line.replace(filename_img, filepath_img)
            lines_total.append(line_new)
    print('\ntotal test: ', len(lines_total))
    
    f = open(filepath_anno_wsi_path_test, 'w')
    f.writelines(lines_total)
    f.close()
    
def create_wsi_patch_train_files():
    """by huanggf"""
    from datasets.our_datasets import WSI_PATCH
    
    dirpath_wsi_path = WSI_PATCH.root_path
    
    dirnames_train = WSI_PATCH.get_default_train_datasets()
    dirnames_test = WSI_PATCH.get_default_test_datasets()
    
    filename_anno = WSI_PATCH.default_anno_file_name
    
    filepath_anno_wsi_path_train = WSI_PATCH.at(WSI_PATCH.default_train_file_path)
    filepath_anno_wsi_path_test = WSI_PATCH.at(WSI_PATCH.default_test_file_path)
    
    lines_total = []
    for dirname_sub in dirnames_train:
        dirpath_sub = '%s/%s' % (dirpath_wsi_path, dirname_sub)
        filepath_anno = os.path.join(dirpath_sub, filename_anno)
        
        f = open(filepath_anno)
        lines = f.readlines()
        f.close()
        print(dirname_sub, len(lines))
        
        for line in lines:
            arrs = line.split(' ')
            filename_img = arrs[0]
            
            filepath_img = '%s/images/%s' % (dirpath_sub, filename_img)
            line_new = line.replace(filename_img, filepath_img)
            if not line_new.endswith('\n'):
                line_new = '%s\n' % (line_new)
            lines_total.append(line_new)
    print('\ntotal train: ', len(lines_total))
    
    f = open(filepath_anno_wsi_path_train, 'w')
    f.writelines(lines_total)
    f.close()
    
    lines_total = []
    for dirname_sub in dirnames_test:
        dirpath_sub = '%s/%s' % (dirpath_wsi_path, dirname_sub)
        filepath_anno = os.path.join(dirpath_sub, filename_anno)
        
        f = open(filepath_anno)
        lines = f.readlines()
        f.close()
        print(dirname_sub, len(lines))
        
        for line in lines:
            arrs = line.split(' ')
            filename_img = arrs[0]
            filepath_img = '%s/images/%s' % (dirpath_sub, filename_img)
            line_new = line.replace(filename_img, filepath_img)
            lines_total.append(line_new)
    print('\ntotal test: ', len(lines_total))
    
    f = open(filepath_anno_wsi_path_test, 'w')
    f.writelines(lines_total)
    f.close()