package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import java.util.Date;
import java.util.Random;
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
import java.util.ArrayList;
import java.util.Properties;

class FileUtil_downloadThirdPartySearchFile_15_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private Query xml;

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
    void testDownloadThirdPartySearchFile_Success() throws Exception {
        String sellerId = "seller1";
        String type = "type1";
        String page = "page1";
        String status = "status1";
        String response = "<xml>response</xml>";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File file = fileUtil.downloadThirdPartySearchFile(sellerId, type, page, status);
        assertNotNull(file);
        assertTrue(file.exists());
        assertTrue(file.length() > 1000);
    }

    @Test
    void testDownloadThirdPartySearchFile_Failure() throws Exception {
        String sellerId = "seller1";
        String type = "type1";
        String page = "page1";
        String status = "status1";
        when(xml.sendRequest(anyString())).thenThrow(new RuntimeException("Simulated exception"));
        File file = fileUtil.downloadThirdPartySearchFile(sellerId, type, page, status);
        assertNull(file);
    }

    @Test
    void testDownloadThirdPartySearchFile_SmallFile() throws Exception {
        String sellerId = "seller1";
        String type = "type1";
        String page = "page1";
        String status = "status1";
        String response = "<xml>short response</xml>";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File file = fileUtil.downloadThirdPartySearchFile(sellerId, type, page, status);
        assertNull(file);
    }
}
