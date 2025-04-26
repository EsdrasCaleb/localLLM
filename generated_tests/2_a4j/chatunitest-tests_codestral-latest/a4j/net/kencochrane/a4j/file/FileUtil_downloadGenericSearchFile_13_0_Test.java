package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import java.util.Date;
import java.util.Random;
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
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class FileUtil_downloadGenericSearchFile_13_0_Test {

    @Mock
    private Query xml;

    @InjectMocks
    private FileUtil fileUtil;

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
    public void testDownloadGenericSearchFile_Success() throws Exception {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        String response = "successful response";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File result = fileUtil.downloadGenericSearchFile(searchType, searchTerm, mode, type, page, offer);
        assertNotNull(result);
        assertTrue(result.exists());
        assertTrue(result.length() > 1000);
    }

    @Test
    public void testDownloadGenericSearchFile_Failure() throws Exception {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        String response = "short response";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File result = fileUtil.downloadGenericSearchFile(searchType, searchTerm, mode, type, page, offer);
        assertNull(result);
    }

    @Test
    public void testDownloadGenericSearchFile_Exception() throws Exception {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        when(xml.sendRequest(anyString())).thenThrow(new RuntimeException("Test Exception"));
        File result = fileUtil.downloadGenericSearchFile(searchType, searchTerm, mode, type, page, offer);
        assertNull(result);
    }
}
