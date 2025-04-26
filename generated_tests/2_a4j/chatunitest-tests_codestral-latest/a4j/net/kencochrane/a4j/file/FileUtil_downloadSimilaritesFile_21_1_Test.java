package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
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
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
public class FileUtil_downloadSimilaritesFile_21_1_Test {

    @Mock
    private Query xml;

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() {
        fileUtil = new FileUtil();
    }

    @Test
    public void testDownloadSimilaritesFile_Success() throws Exception {
        String asin = "B001234567";
        String page = "1";
        String saveFileName = "similarities.txt";
        String response = "dummy response data";
        when(xml.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadSimilaritesFile(asin, page, saveFileName);
        assertTrue(result);
        File file = new File(saveFileName);
        assertTrue(file.exists());
        assertTrue(file.length() > 1000);
        file.delete();
    }

    @Test
    public void testDownloadSimilaritesFile_Failure_FileSizeTooSmall() throws Exception {
        String asin = "B001234567";
        String page = "1";
        String saveFileName = "similarities.txt";
        String response = "small response";
        when(xml.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadSimilaritesFile(asin, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertTrue(file.exists());
        assertTrue(file.length() < 1000);
        file.delete();
    }

    @Test
    public void testDownloadSimilaritesFile_Failure_Exception() throws Exception {
        String asin = "B001234567";
        String page = "1";
        String saveFileName = "similarities.txt";
        when(xml.sendRequest(anyString())).thenThrow(new RuntimeException("Simulated exception"));
        boolean result = fileUtil.downloadSimilaritesFile(asin, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertFalse(file.exists());
    }
}
