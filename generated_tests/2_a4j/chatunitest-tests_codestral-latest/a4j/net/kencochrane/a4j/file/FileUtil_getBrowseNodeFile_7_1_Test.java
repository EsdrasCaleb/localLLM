package net.kencochrane.a4j.file;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
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
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
public class FileUtil_getBrowseNodeFile_7_1_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @Mock
    private File cachedFile;

    @Mock
    private File tempFile;

    @BeforeEach
    public void setUp() throws NoSuchFieldException, IllegalAccessException {
        Field cacheDirField = FileUtil.class.getDeclaredField("cacheDir");
        cacheDirField.setAccessible(true);
        cacheDirField.set(fileUtil, "/cache/");
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        oldestAgeField.set(fileUtil, 1000L);
    }

    @Test
    public void testGetBrowseNodeFile_CachedFileNotExists_DownloadFail() throws IOException {
        String mode = "mode";
        String node = "node";
        String page = "page";
        String filename = mode + "_" + node + "_" + page + ".xml";
    }
}
