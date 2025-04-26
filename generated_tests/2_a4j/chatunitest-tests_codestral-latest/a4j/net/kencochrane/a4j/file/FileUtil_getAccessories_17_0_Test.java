package net.kencochrane.a4j.file;

import java.io.File;
import java.lang.reflect.Field;
import java.util.ArrayList;
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
import java.util.Date;
import java.util.Properties;
import java.util.Random;

class FileUtil_getAccessories_17_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private File mockCachedFile;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field cacheDirField = FileUtil.class.getDeclaredField("cacheDir");
        cacheDirField.setAccessible(true);
        cacheDirField.set(fileUtil, "testCacheDir/");
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        oldestAgeField.set(fileUtil, 1000L);
    }

    @Test
    void testGetAccessories_CachedFileExistsAndIsGood() throws Exception {
        String asin = "12345";
        ArrayList<String> asins = new ArrayList<>();
        when(mockCachedFile.exists()).thenReturn(true);
        when(mockCachedFile.lastModified()).thenReturn(System.currentTimeMillis());
        File result = fileUtil.getAccessories(asin, asins);
        assertNotNull(result);
        assertEquals("testCacheDir/A_12345.XML", result.getPath());
    }

    @Test
    void testGetAccessories_CachedFileExistsButIsOld() throws Exception {
        String asin = "12345";
        ArrayList<String> asins = new ArrayList<>();
        when(mockCachedFile.exists()).thenReturn(true);
        when(mockCachedFile.lastModified()).thenReturn(0L);
        File result = fileUtil.getAccessories(asin, asins);
        assertNotNull(result);
        assertEquals("testCacheDir/A_12345.XML", result.getPath());
    }

    @Test
    void testGetAccessories_CachedFileDoesNotExist() throws Exception {
        String asin = "12345";
        ArrayList<String> asins = new ArrayList<>();
        when(mockCachedFile.exists()).thenReturn(false);
        File result = fileUtil.getAccessories(asin, asins);
        assertNotNull(result);
        assertEquals("testCacheDir/A_12345.XML", result.getPath());
    }

    @Test
    void testGetAccessories_DownloadFails() throws Exception {
        String asin = "12345";
        ArrayList<String> asins = new ArrayList<>();
        when(mockCachedFile.exists()).thenReturn(false);
        File result = fileUtil.getAccessories(asin, asins);
        assertNull(result);
    }
}
