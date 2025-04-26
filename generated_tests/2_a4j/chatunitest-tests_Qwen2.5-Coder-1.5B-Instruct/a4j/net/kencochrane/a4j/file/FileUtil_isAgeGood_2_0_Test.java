package net.kencochrane.a4j.file;

import java.io.File;
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

    @Test
    public void testIsAgeGoodWithOldFile() throws Exception {
        // Create an instance of FileUtil
        FileUtil fileUtil = new FileUtil();
        // Mock the file object
        File mockFile = mock(File.class);
        // Simulate a very old file
        when(mockFile.lastModified()).thenReturn(0L);
        // Call the method under test
        boolean result = fileUtil.isAgeGood(mockFile);
        // Verify the result
        assertFalse(result);
    }

    @Test
    public void testIsAgeGoodWithRecentFile() throws Exception {
        // Create an instance of FileUtil
        FileUtil fileUtil = new FileUtil();
        // Mock the file object
        File mockFile = mock(File.class);
        // Simulate a recent file
        when(mockFile.lastModified()).thenReturn(System.currentTimeMillis() - 86400000L);
        // Call the method under test
        boolean result = fileUtil.isAgeGood(mockFile);
        // Verify the result
        assertTrue(result);
    }

    @Test
    public void testIsAgeGoodWithSmallFile() throws Exception {
        // Create an instance of FileUtil
        FileUtil fileUtil = new FileUtil();
        // Mock the file object
        File mockFile = mock(File.class);
        // Simulate a very old file
        when(mockFile.lastModified()).thenReturn(0L);
        // Simulate a small file
        when(mockFile.length()).thenReturn(999L);
        // Call the method under test
        boolean result = fileUtil.isAgeGood(mockFile);
        // Verify the result
        assertFalse(result);
    }

    @Test
    public void testIsAgeGoodWithNullFile() throws Exception {
        // Create an instance of FileUtil
        FileUtil fileUtil = new FileUtil();
        // Mock the file object
        File mockFile = mock(File.class);
        // Simulate a very old file
        when(mockFile.lastModified()).thenReturn(0L);
        // Simulate a small file
        when(mockFile.length()).thenReturn(1000L);
        // Call the method under test
        boolean result = fileUtil.isAgeGood(null);
        // Verify the result
        assertFalse(result);
    }
}
