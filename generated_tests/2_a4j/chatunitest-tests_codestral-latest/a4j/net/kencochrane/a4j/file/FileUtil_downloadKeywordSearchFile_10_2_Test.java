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
public class FileUtil_downloadKeywordSearchFile_10_2_Test {

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
    public void testDownloadKeywordSearchFile_Success() throws Exception {
        String searchTerm = "testTerm";
        String productLine = "testLine";
        String type = "testType";
        String page = "testPage";
        String response = "testResponse";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File file = fileUtil.downloadKeywordSearchFile(searchTerm, productLine, type, page);
        assertNotNull(file);
        assertTrue(file.exists());
        assertTrue(file.length() > 0);
    }

    @Test
    public void testDownloadKeywordSearchFile_Failure() throws Exception {
        String searchTerm = "testTerm";
        String productLine = "testLine";
        String type = "testType";
        String page = "testPage";
        String response = "short";
        when(xml.sendRequest(anyString())).thenReturn(response);
        File file = fileUtil.downloadKeywordSearchFile(searchTerm, productLine, type, page);
        assertNull(file);
    }

    @Test
    public void testDownloadKeywordSearchFile_Exception() throws Exception {
        String searchTerm = "testTerm";
        String productLine = "testLine";
        String type = "testType";
        String page = "testPage";
        when(xml.sendRequest(anyString())).thenThrow(new RuntimeException("Test Exception"));
        File file = fileUtil.downloadKeywordSearchFile(searchTerm, productLine, type, page);
        assertNull(file);
    }
}
