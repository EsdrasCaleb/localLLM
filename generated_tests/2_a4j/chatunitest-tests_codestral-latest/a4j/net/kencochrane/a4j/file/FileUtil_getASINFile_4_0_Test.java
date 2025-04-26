package net.kencochrane.a4j.file;

import java.io.File;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
public class FileUtil_getASINFile_4_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private File mockFile;

    @BeforeEach
    public void setUp() throws Exception {
        Field cacheDirField = FileUtil.class.getDeclaredField("cacheDir");
        cacheDirField.setAccessible(true);
        cacheDirField.set(fileUtil, "testCacheDir/");
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        oldestAgeField.set(fileUtil, 1000L);
    }

    @Test
    public void testGetASINFile_CachedFileExistsAndIsGood() throws Exception {
        String asin = "ASIN123";
        String type = "type";
        String offer = "offer";
        String page = "page";
        when(mockFile.exists()).thenReturn(true);
        when(mockFile.lastModified()).thenReturn(System.currentTimeMillis());
        File result = fileUtil.getASINFile(asin, type, offer, page);
        assertNotNull(result);
        assertEquals("testCacheDir/ASIN123_OFFER_TYPE_PAGE.XML", result.getPath());
    }

    @Test
    public void testGetASINFile_CachedFileExistsButIsOld() throws Exception {
        String asin = "ASIN123";
        String type = "type";
        String offer = "offer";
        String page = "page";
        when(mockFile.exists()).thenReturn(true);
        when(mockFile.lastModified()).thenReturn(System.currentTimeMillis() - 2000L);
        File result = fileUtil.getASINFile(asin, type, offer, page);
        assertNotNull(result);
        assertEquals("testCacheDir/ASIN123_OFFER_TYPE_PAGE.XML", result.getPath());
    }

    @Test
    public void testGetASINFile_CachedFileDoesNotExist() throws Exception {
        String asin = "ASIN123";
        String type = "type";
        String offer = "offer";
        String page = "page";
        when(mockFile.exists()).thenReturn(false);
        File result = fileUtil.getASINFile(asin, type, offer, page);
        assertNull(result);
    }

    @Test
    public void testGetASINFile_DownloadFails() throws Exception {
        String asin = "ASIN123";
        String type = "type";
        String offer = "offer";
        String page = "page";
        when(mockFile.exists()).thenReturn(true);
        when(mockFile.lastModified()).thenReturn(System.currentTimeMillis() - 2000L);
        File result = fileUtil.getASINFile(asin, type, offer, page);
        assertNotNull(result);
        assertEquals("testCacheDir/ASIN123_OFFER_TYPE_PAGE.XML", result.getPath());
    }
}
