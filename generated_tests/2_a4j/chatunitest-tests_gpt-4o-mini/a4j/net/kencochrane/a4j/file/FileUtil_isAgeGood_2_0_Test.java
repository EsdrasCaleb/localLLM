package net.kencochrane.a4j.file;

import java.io.File;
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

public class FileUtil_isAgeGood_2_0_Test {

    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() {
        fileUtil = new FileUtil();
    }

    @Test
    public void testIsAgeGood_NullFile() {
        assertFalse(fileUtil.isAgeGood(null));
    }

    @Test
    public void testIsAgeGood_SmallFile() {
        File smallFile = mock(File.class);
        // Size < 1000
        when(smallFile.length()).thenReturn(500L);
        assertFalse(fileUtil.isAgeGood(smallFile));
    }

    @Test
    public void testIsAgeGood_YoungFile() throws Exception {
        File youngFile = mock(File.class);
        // Size >= 1000
        when(youngFile.length()).thenReturn(1500L);
        // Current time
        when(youngFile.lastModified()).thenReturn(System.currentTimeMillis());
        // Set oldestAge to 5000 milliseconds
        setOldestAge(fileUtil, 5000L);
        assertTrue(fileUtil.isAgeGood(youngFile));
    }

    @Test
    public void testIsAgeGood_OldFile() throws Exception {
        File oldFile = mock(File.class);
        // Size >= 1000
        when(oldFile.length()).thenReturn(1500L);
        // Modified 10 seconds ago
        when(oldFile.lastModified()).thenReturn(System.currentTimeMillis() - 10000);
        // Set oldestAge to 5000 milliseconds
        setOldestAge(fileUtil, 5000L);
        assertFalse(fileUtil.isAgeGood(oldFile));
    }

    private void setOldestAge(FileUtil fileUtil, long age) throws Exception {
        Field field = FileUtil.class.getDeclaredField("oldestAge");
        field.setAccessible(true);
        field.setLong(fileUtil, age);
    }
}
