package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
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
public class FileUtil_downloadBlendedSearchFile_9_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private Query xml;

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
    public void testDownloadBlendedSearchFileSuccess() throws Exception {
        String searchTerm = "testTerm";
        String type = "testType";
        String response = "testResponse";
        when(xml.sendRequest(anyString())).thenReturn(response);
        when(xml.BlendedSearchGenerator(type, searchTerm)).thenReturn("testRequest");
        File result = fileUtil.downloadBlendedSearchFile(searchTerm, type);
        assertNotNull(result);
        assertTrue(result.length() > 1000);
    }

    @Test
    public void testDownloadBlendedSearchFileFailure() throws Exception {
        String searchTerm = "testTerm";
        String type = "testType";
        when(xml.sendRequest(anyString())).thenThrow(new IOException("Test Exception"));
        File result = fileUtil.downloadBlendedSearchFile(searchTerm, type);
        assertNull(result);
    }

    @Test
    public void testDownloadBlendedSearchFileInvalidSize() throws Exception {
        String searchTerm = "testTerm";
        String type = "testType";
        String response = "shortResponse";
        when(xml.sendRequest(anyString())).thenReturn(response);
        when(xml.BlendedSearchGenerator(type, searchTerm)).thenReturn("testRequest");
        File result = fileUtil.downloadBlendedSearchFile(searchTerm, type);
        assertNull(result);
    }
}
