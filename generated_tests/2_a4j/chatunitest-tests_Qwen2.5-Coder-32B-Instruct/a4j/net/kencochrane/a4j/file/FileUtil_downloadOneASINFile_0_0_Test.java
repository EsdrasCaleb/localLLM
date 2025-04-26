package net.kencochrane.a4j.file;

import static org.mockito.ArgumentMatchers.*;
import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Date;
import java.util.Properties;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class FileUtil_downloadOneASINFile_0_0_Test {

    @Mock
    private Query mockQuery;

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testDownloadOneASINFile_Success() throws Exception {
        String asin = "B08N5WRWNW";
        String type = "type";
        String offer = "offer";
        String page = "1";
        String saveFileName = "testFile.txt";
        String mockResponse = "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met. " + "This is a mock response with more than 1000 bytes to ensure the file size condition is met.";
        when(mockQuery.queryGenerator(anyString(), anyString(), anyString(), anyString(), any(ArrayList.class))).thenReturn("mockQuery");
        when(mockQuery.sendRequest(anyString())).thenReturn(mockResponse);
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertTrue(result);
        File file = new File(saveFileName);
        assertTrue(file.exists() && file.length() >= 1000);
        file.delete();
    }

    @Test
    public void testDownloadOneASINFile_Failure_SmallFileSize() throws Exception {
        String asin = "B08N5WRWNW";
        String type = "type";
        String offer = "offer";
        String page = "1";
        String saveFileName = "testFile.txt";
        String mockResponse = "Short response";
        when(mockQuery.queryGenerator(anyString(), anyString(), anyString(), anyString(), any(ArrayList.class))).thenReturn("mockQuery");
        when(mockQuery.sendRequest(anyString())).thenReturn(mockResponse);
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertTrue(file.exists() && file.length() < 1000);
        file.delete();
    }

    @Test
    public void testDownloadOneASINFile_Failure_Exception() throws Exception {
        String asin = "B08N5WRWNW";
        String type = "type";
        String offer = "offer";
        String page = "1";
        String saveFileName = "testFile.txt";
        when(mockQuery.queryGenerator(anyString(), anyString(), anyString(), anyString(), any(ArrayList.class))).thenReturn("mockQuery");
        when(mockQuery.sendRequest(anyString())).thenThrow(new Exception("Mock Exception"));
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertFalse(file.exists());
    }
}
