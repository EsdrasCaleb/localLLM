package net.kencochrane.a4j.file;

import java.io.File;
import java.time.Instant;
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

class FileUtil_isAgeGood_2_0_Test {

    private FileUtil fileUtil;

    @BeforeEach
    void setUp() {
        fileUtil = new FileUtil();
        fileUtil.oldestAge = 1000;
    }

    @Test
    void isAgeGood_NullFile_ReturnsFalse() {
        assertFalse(fileUtil.isAgeGood(null));
    }

    @Test
    void isAgeGood_FileSizeLessThan1000_ReturnsFalse() {
        File file = mock(File.class);
        Mockito.when(file.length()).thenReturn(500L);
        assertFalse(fileUtil.isAgeGood(file));
    }

    @Test
    void isAgeGood_FileAgeOlderThanOldestAge_ReturnsFalse() {
        File file = mock(File.class);
        Mockito.when(file.lastModified()).thenReturn(Instant.now().minusSeconds(2000).getEpochSecond());
        Mockito.when(file.length()).thenReturn(2000L);
        assertFalse(fileUtil.isAgeGood(file));
    }

    @Test
    void isAgeGood_FileAgeYoungerThanOldestAge_ReturnsTrue() {
        File file = mock(File.class);
        Mockito.when(file.lastModified()).thenReturn(Instant.now().minusSeconds(500).getEpochSecond());
        Mockito.when(file.length()).thenReturn(2000L);
        assertTrue(fileUtil.isAgeGood(file));
    }
}
