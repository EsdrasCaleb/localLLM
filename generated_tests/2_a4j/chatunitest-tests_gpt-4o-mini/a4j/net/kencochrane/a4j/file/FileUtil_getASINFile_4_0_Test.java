package net.kencochrane.a4j.file;

import java.io.File;
import static org.mockito.ArgumentMatchers.anyString;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

class FileUtil_getASINFile_4_0_Test {

    private FileUtil fileUtil;

    private static final String CACHE_DIR = "/cache/";

    @BeforeEach
    void setUp() {
        fileUtil = Mockito.spy(new FileUtil());
        fileUtil.cacheDir = CACHE_DIR;
    }

    @Test
    void testGetASINFile_CachedFileDoesNotExistAndDownloadFails() {
        String asin = "B000123456";
        String type = "type1";
        String offer = "offer1";
        String page = "1";
        when(fileUtil.downloadOneASINFile(asin, type, offer, page, "t_" + asin + "_" + offer + "_" + type + "_" + page + ".xml")).thenReturn(false);
        assertNull(fileUtil.getASINFile(asin, type, offer, page));
    }
}
