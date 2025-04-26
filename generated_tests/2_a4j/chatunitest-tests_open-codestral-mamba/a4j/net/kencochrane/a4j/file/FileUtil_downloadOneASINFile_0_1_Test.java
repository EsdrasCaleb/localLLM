package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Field;
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

public class FileUtil_downloadOneASINFile_0_1_Test {

    @Mock
    private Query xml;

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.initMocks(this);
        Field cacheDirField = FileUtil.class.getDeclaredField("cacheDir");
        cacheDirField.setAccessible(true);
        cacheDirField.set(fileUtil, "testCacheDir");
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        oldestAgeField.set(fileUtil, 1000L);
    }

    @Test
    public void testDownloadOneASINFile() throws Exception {
        String asin = "testAsin";
        String type = "testType";
        String offer = "testOffer";
        String page = "testPage";
        String saveFileName = "testSaveFileName";
        when(xml.sendRequest(anyString())).thenReturn("dummy response");
        boolean downloaded = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        File file = new File(saveFileName);
        assertTrue(downloaded);
        assertTrue(file.exists());
        assertTrue(file.length() >= 1000);
        file.delete();
    }
}
