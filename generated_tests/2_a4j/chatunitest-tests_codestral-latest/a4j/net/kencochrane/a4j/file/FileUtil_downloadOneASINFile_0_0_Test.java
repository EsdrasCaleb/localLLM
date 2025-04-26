package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
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
public class FileUtil_downloadOneASINFile_0_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private Query xml;

    @BeforeEach
    public void setUp() throws Exception {
        Field cacheDirField = FileUtil.class.getDeclaredField("cacheDir");
        cacheDirField.setAccessible(true);
        cacheDirField.set(fileUtil, "testCacheDir");
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        oldestAgeField.set(fileUtil, 1000L);
    }

    @Test
    public void testDownloadOneASINFileSuccess() throws Exception {
        String asin = "testASIN";
        String type = "testType";
        String offer = "testOffer";
        String page = "testPage";
        String saveFileName = "testFile.txt";
        String response = "test response data";
        when(xml.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertTrue(result);
        File file = new File(saveFileName);
        assertTrue(file.exists());
        assertTrue(file.length() > 1000);
    }

    @Test
    public void testDownloadOneASINFileFailure() throws Exception {
        String asin = "testASIN";
        String type = "testType";
        String offer = "testOffer";
        String page = "testPage";
        String saveFileName = "testFile.txt";
        String response = "short response";
        when(xml.sendRequest(anyString())).thenReturn(response);
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertTrue(file.exists());
        assertTrue(file.length() < 1000);
    }

    @Test
    public void testDownloadOneASINFileException() throws Exception {
        String asin = "testASIN";
        String type = "testType";
        String offer = "testOffer";
        String page = "testPage";
        String saveFileName = "testFile.txt";
        when(xml.sendRequest(anyString())).thenThrow(new RuntimeException("Test Exception"));
        boolean result = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertFalse(result);
        File file = new File(saveFileName);
        assertFalse(file.exists());
    }
}
