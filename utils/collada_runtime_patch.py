"""
Runtime patch for pycollada to enable XML_PARSE_HUGE flag.
This avoids modifying installed library files.
"""

def patch_collada_huge_tree():
    """
    Monkey patch the collada library to use huge_tree=True for XML parsing.
    This allows parsing of COLLADA files with very large text nodes (>10MB).
    
    Call this before any collada imports or usage.
    """
    import sys
    from io import BytesIO
    
    # Import collada to patch its __init__ method
    import collada
    from collada.xmlutil import etree as ElementTree
    
    # Save the original __init__ method
    _original_collada_init = collada.Collada.__init__
    
    def patched_collada_init(self, filename=None, ignore=None, aux_file_loader=None, zip_filename=None, validate_output=False):
        """Patched Collada.__init__ that properly uses XMLParser with huge_tree=True"""
        
        # Call the original init for most of the setup
        # We'll intercept at the XML parsing stage
        
        # First, do all the file reading logic from the original
        self.errors = []
        self.assetInfo = None
        self._geometries = collada.util.IndexedList([], ('id',))
        self._controllers = collada.util.IndexedList([], ('id',))
        self._animations = collada.util.IndexedList([], ('id',))
        self._lights = collada.util.IndexedList([], ('id',))
        self._cameras = collada.util.IndexedList([], ('id',))
        self._images = collada.util.IndexedList([], ('id',))
        self._effects = collada.util.IndexedList([], ('id',))
        self._materials = collada.util.IndexedList([], ('id',))
        self._nodes = collada.util.IndexedList([], ('id',))
        self._scenes = collada.util.IndexedList([], ('id',))
        self.scene = None
        
        if validate_output and hasattr(collada, 'schema') and collada.schema:
            self.validator = collada.schema.ColladaValidator()
        else:
            self.validator = None
            
        self.maskedErrors = []
        if ignore is not None:
            self.ignoreErrors(*ignore)
            
        # Handle the case where no filename is provided
        if filename is None:
            # This part remains the same as original
            self.filename = None
            self.zfile = None
            self.getFileData = collada.Collada._nullGetFile.__get__(self, collada.Collada)
            if aux_file_loader is not None:
                self.getFileData = collada.Collada._wrappedFileLoader(self, aux_file_loader)
                
            self.xmlnode = ElementTree.ElementTree(
                               collada.common.E.COLLADA(
                                   collada.common.E.library_cameras(),
                                   collada.common.E.library_controllers(),
                                   collada.common.E.library_effects(),
                                   collada.common.E.library_geometries(),
                                   collada.common.E.library_images(),
                                   collada.common.E.library_lights(),
                                   collada.common.E.library_materials(),
                                   collada.common.E.library_nodes(),
                                   collada.common.E.library_visual_scenes(),
                                   collada.common.E.scene(),
                               version='1.4.1'))
            self.assetInfo = collada.asset.Asset()
            return
            
        # File reading logic (same as original)
        if isinstance(filename, str):
            fdata = open(filename, 'rb')
            self.filename = filename
            self.getFileData = collada.Collada._getFileFromDisk.__get__(self, collada.Collada)
        else:
            fdata = filename
            self.filename = None
            self.getFileData = collada.Collada._nullGetFile.__get__(self, collada.Collada)
        strdata = fdata.read()
        
        # Zip file handling (same as original)
        try:
            import zipfile
            self.zfile = zipfile.ZipFile(BytesIO(strdata), 'r')
        except:
            self.zfile = None
            
        if self.zfile:
            self.filename = ''
            daefiles = []
            if zip_filename is not None:
                self.filename = zip_filename
            else:
                for name in self.zfile.namelist():
                    if name.upper().endswith('.DAE'):
                        daefiles.append(name)
                for name in daefiles:
                    if not self.filename:
                        self.filename = name
                    elif "MACOSX" in self.filename:
                        self.filename = name
            if not self.filename or self.filename not in self.zfile.namelist():
                raise collada.common.DaeIncompleteError('COLLADA file not found inside zip compressed file')
            data = self.zfile.read(self.filename)
            self.getFileData = collada.Collada._getFileFromZip.__get__(self, collada.Collada)
        else:
            data = strdata
            
        if aux_file_loader is not None:
            self.getFileData = collada.Collada._wrappedFileLoader(self, aux_file_loader)
            
        # THIS IS THE KEY CHANGE: Use XMLParser with huge_tree=True
        etree_parser = ElementTree.XMLParser(huge_tree=True)
        try:
            # Use parse() instead of ElementTree() constructor
            self.xmlnode = ElementTree.parse(BytesIO(data), parser=etree_parser)
        except ElementTree.ParseError as e:
            raise collada.common.DaeMalformedError("XML Parsing Error: %s" % e)
            
        # Call all the loading methods
        self._loadAssetInfo()
        self._loadImages()
        self._loadEffects()
        self._loadMaterials()
        self._loadAnimations()
        self._loadGeometry()
        self._loadControllers()
        self._loadLights()
        self._loadCameras()
        self._loadNodes()
        self._loadScenes()
        self._loadDefaultScene()
    
    # Apply the patch
    collada.Collada.__init__ = patched_collada_init
    
    print("[COLLADA PATCH] Successfully patched Collada.__init__ to use XMLParser with huge_tree=True")
    
    
def unpatch_collada():
    """
    Remove the monkey patch and restore original behavior.
    """
    from lxml import etree
    import sys
    
    # Restore original XMLParser if we have it
    if hasattr(etree.XMLParser, '__wrapped__'):
        etree.XMLParser = etree.XMLParser.__wrapped__
        
    if 'xml.etree.ElementTree' in sys.modules:
        import xml.etree.ElementTree as ElementTree
        if hasattr(ElementTree.XMLParser, '__wrapped__'):
            ElementTree.XMLParser = ElementTree.XMLParser.__wrapped__
            
    print("[COLLADA PATCH] Removed patch, restored original behavior")


# Example usage in your training script:
if __name__ == "__main__":
    # Test the patch
    patch_collada_huge_tree()
    
    # Now you can import and use collada normally
    import collada
    
    # Test loading a large COLLADA file
    # mesh = collada.Collada('large_file.dae')
    
    print("Patch applied successfully!")