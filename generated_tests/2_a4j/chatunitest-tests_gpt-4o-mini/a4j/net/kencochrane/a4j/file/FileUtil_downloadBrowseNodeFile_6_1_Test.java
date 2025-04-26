package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
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
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_downloadBrowseNodeFile_6_1_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private Query queryMock;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testDownloadBrowseNodeFile_SuccessfulDownload() throws Exception {
        String mode = "testMode";
        String node = "testNode";
        String page = "1";
        String saveFileName = "testFile.txt";
        String response = "sample response content";
        when(queryMock.browseNodeQueryGenerator(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn("queryString");
        when(queryMock.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadBrowseNodeFile(mode, node, page, saveFileName);
        assertTrue(result);
        assertTrue(new File(saveFileName).exists());
        assertTrue(new File(saveFileName).length() >= 1000);
    }

    @Test
    public void testDownloadBrowseNodeFile_FileTooSmall() throws Exception {
        String mode = "testMode";
        String node = "testNode";
        String page = "1";
        String saveFileName = "testFile.txt";
        String response = "small";
        when(queryMock.browseNodeQueryGenerator(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn("queryString");
        when(queryMock.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadBrowseNodeFile(mode, node, page, saveFileName);
        assertFalse(result);
        assertTrue(new File(saveFileName).exists());
        assertTrue(new File(saveFileName).length() < 1000);
    }

    @Test
    public void testDownloadBrowseNodeFile_ExceptionHandling() throws Exception {
        String mode = "testMode";
        String node = "testNode";
        String page = "1";
        String saveFileName = "testFile.txt";
        when(queryMock.browseNodeQueryGenerator(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn("queryString");
        when(queryMock.sendRequest(anyString())).thenThrow(new IOException("Network error"));
        boolean result = fileUtil.downloadBrowseNodeFile(mode, node, page, saveFileName);
        assertFalse(result);
    }

    @Test
    public void testDownloadBrowseNodeFile_FileNotCreated() throws Exception {
        String mode = "testMode";
        String node = "testNode";
        String page = "1";
        String saveFileName = "nonExistentDirectory/testFile.txt";
        String response = "sample response content";
        when(queryMock.browseNodeQueryGenerator(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn("queryString");
        when(queryMock.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadBrowseNodeFile(mode, node, page, saveFileName);
        assertFalse(result);
        assertFalse(new File(saveFileName).exists());
    }
}
