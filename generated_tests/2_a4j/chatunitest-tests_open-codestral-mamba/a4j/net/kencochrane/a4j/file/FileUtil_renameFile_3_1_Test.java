package net.kencochrane.a4j.file;

import java.io.File;
import java.io.IOException;
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
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_renameFile_3_1_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.initMocks(this);
        Field field = FileUtil.class.getDeclaredField("cacheDir");
        field.setAccessible(true);
        field.set(fileUtil, "testCacheDir");
        field = FileUtil.class.getDeclaredField("oldestAge");
        field.setAccessible(true);
        field.set(fileUtil, 1000L);
    }

    @Test
    public void testRenameFile() throws IOException {
        String oldFileName = "oldFile.txt";
        String newFileName = "newFile.txt";
        File oldFile = mock(File.class);
        File newFile = mock(File.class);
        when(oldFile.exists()).thenReturn(true);
        when(oldFile.renameTo(newFile)).thenReturn(true);
        fileUtil.renameFile(oldFileName, newFileName);
        assertFalse(oldFile.exists());
        assertTrue(newFile.exists());
    }
}
