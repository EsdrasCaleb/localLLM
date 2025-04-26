package net.kencochrane.a4j.file;

import java.util.Date;
import java.io.File;
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

public class FileUtil_isAgeGood_2_0_Test {

    private FileUtil fileUtil;

    @Mock
    private File mockFile;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        fileUtil = new FileUtil();
    }

    @Test
    @DisplayName("Test isAgeGood with valid file")
    public void testIsAgeGood_ValidFile() {
        // Given
        long currentTime = new Date().getTime();
        // 10kB
        long fileAge = 10000;
        long oldestAge = currentTime - fileAge;
        when(mockFile.lastModified()).thenReturn(fileAge);
        when(mockFile.length()).thenReturn(10000L);
        // When
        boolean result = fileUtil.isAgeGood(mockFile);
        // Then
        assertEquals(true, result);
    }

    @Test
    @DisplayName("Test isAgeGood with file age less than oldest age")
    public void testIsAgeGood_FileAgeLessThanOldestAge() {
        // Given
        long currentTime = new Date().getTime();
        long fileAge = currentTime - (fileUtil.oldestAge + 1);
        when(mockFile.lastModified()).thenReturn(fileAge);
        when(mockFile.length()).thenReturn(10000L);
        // When
        boolean result = fileUtil.isAgeGood(mockFile);
        // Then
        assertEquals(false, result);
    }

    @Test
    @DisplayName("Test isAgeGood with file age equal to oldest age")
    public void testIsAgeGood_FileAgeEqualToOldestAge() {
        // Given
        long currentTime = new Date().getTime();
        long fileAge = currentTime - fileUtil.oldestAge;
        when(mockFile.lastModified()).thenReturn(fileAge);
        when(mockFile.length()).thenReturn(10000L);
        // When
        boolean result = fileUtil.isAgeGood(mockFile);
        // Then
        assertEquals(true, result);
    }

    @Test
    @DisplayName("Test isAgeGood with file length less than 1000")
    public void testIsAgeGood_FileLengthLessThan1000() {
        // Given
        when(mockFile.lastModified()).thenReturn(1000L);
        when(mockFile.length()).thenReturn(999L);
        // When
        boolean result = fileUtil.isAgeGood(mockFile);
        // Then
        assertEquals(false, result);
    }

    @Test
    @DisplayName("Test isAgeGood with null file")
    public void testIsAgeGood_NullFile() {
        // Given
        // When
        boolean result = fileUtil.isAgeGood(null);
        // Then
        assertEquals(false, result);
    }
}
