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
public class FileUtil_downloadCart_23_0_Test {

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
    public void testDownloadCartSuccess() throws Exception {
        String cartQuery = "testQuery";
        String response = "testResponse";
        when(xml.sendRequest(cartQuery)).thenReturn(response);
        File result = fileUtil.downloadCart(cartQuery);
        assertNotNull(result);
        assertTrue(result.exists());
        assertTrue(result.length() > 1000);
    }

    @Test
    public void testDownloadCartFailure() throws Exception {
        String cartQuery = "testQuery";
        when(xml.sendRequest(cartQuery)).thenThrow(new RuntimeException("Test Exception"));
        File result = fileUtil.downloadCart(cartQuery);
        assertNull(result);
    }

    @Test
    public void testDownloadCartInvalidFileSize() throws Exception {
        String cartQuery = "testQuery";
        String response = "testResponse";
        when(xml.sendRequest(cartQuery)).thenReturn(response);
        File result = fileUtil.downloadCart(cartQuery);
        assertNull(result);
    }
}
