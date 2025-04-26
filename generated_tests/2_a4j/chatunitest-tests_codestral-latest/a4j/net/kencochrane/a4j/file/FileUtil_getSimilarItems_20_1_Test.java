package net.kencochrane.a4j.file;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
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
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_getSimilarItems_20_1_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() throws IOException {
        MockitoAnnotations.openMocks(this);
        fileUtil.cacheDir = "testCache/";
        // 10 seconds
        fileUtil.oldestAge = 10000;
        // Create cache directory if it doesn't exist
        File cacheDir = new File(fileUtil.cacheDir);
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
    }

    @Test
    public void testGetSimilarItems_CachedFileExistsAndIsFresh() throws Exception {
        String asin = "12345";
        String page = "1";
        String cachedFileName = fileUtil.cacheDir + "S_" + asin + ".XML";
        File cachedFile = new File(cachedFileName);
        // Create a fresh cached file
        cachedFile.createNewFile();
        touch(cachedFile);
        File result = fileUtil.getSimilarItems(asin, page);
        assertNotNull(result);
        assertEquals(cachedFile, result);
    }

    private void touch(File file) throws Exception {
        Method method = File.class.getDeclaredMethod("setLastModified", long.class);
        method.setAccessible(true);
        method.invoke(file, System.currentTimeMillis());
    }
}
