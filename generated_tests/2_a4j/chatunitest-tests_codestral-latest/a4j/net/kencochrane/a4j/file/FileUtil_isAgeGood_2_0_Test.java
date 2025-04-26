package net.kencochrane.a4j.file;

import java.io.File;
import java.lang.reflect.Field;
import java.util.Date;
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
import java.util.Properties;
import java.util.Random;

class FileUtil_isAgeGood_2_0_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Set the oldestAge field using reflection
        Field oldestAgeField = FileUtil.class.getDeclaredField("oldestAge");
        oldestAgeField.setAccessible(true);
        // Set oldestAge to 10 seconds for testing
        oldestAgeField.set(fileUtil, 10000L);
    }

    @Test
    void testIsAgeGood_FileIsNull() {
        assertFalse(fileUtil.isAgeGood(null));
    }

    @Test
    void testIsAgeGood_FileSizeLessThan1000() throws Exception {
        File mockFile = mock(File.class);
        when(mockFile.length()).thenReturn(999L);
        assertFalse(fileUtil.isAgeGood(mockFile));
    }

    @Test
    void testIsAgeGood_FileAgeWithinLimit() throws Exception {
        File mockFile = mock(File.class);
        when(mockFile.length()).thenReturn(1000L);
        // 5 seconds old
        when(mockFile.lastModified()).thenReturn(new Date().getTime() - 5000L);
        assertTrue(fileUtil.isAgeGood(mockFile));
    }

    @Test
    void testIsAgeGood_FileAgeExceedsLimit() throws Exception {
        File mockFile = mock(File.class);
        when(mockFile.length()).thenReturn(1000L);
        // 15 seconds old
        when(mockFile.lastModified()).thenReturn(new Date().getTime() - 15000L);
        assertFalse(fileUtil.isAgeGood(mockFile));
    }
}
