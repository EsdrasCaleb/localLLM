package net.kencochrane.a4j.file;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;
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
public class FileUtil_isAgeGood_2_1_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil fileUtilMocks;

    @Test
    public void testIsAgeGood() throws IOException, InterruptedException {
        // Create a mock file
        File file = new File("mockFile.txt");
        file.createNewFile();
        when(fileUtilMocks.isAgeGood(file)).thenReturn(true);
        // Assert that the method returns true
        assertTrue(fileUtilMocks.isAgeGood(file));
    }

    @Test
    public void testIsAgeGood_BadFile() throws IOException, InterruptedException {
        // Create a mock file
        File file = new File("mockFile.txt");
        file.createNewFile();
        when(fileUtilMocks.isAgeGood(file)).thenReturn(false);
        // Assert that the method returns false
        assertFalse(fileUtilMocks.isAgeGood(file));
    }

    @Test
    public void testIsAgeGood_NullFile() throws IOException, InterruptedException {
        // Assert that the method returns false when given a null file
        assertFalse(fileUtilMocks.isAgeGood(null));
    }

    @Test
    public void testIsAgeGood_FileTooOld() throws IOException, InterruptedException {
        // Create a mock file
        File file = new File("mockFile.txt");
        file.createNewFile();
        when(fileUtilMocks.isAgeGood(file)).thenReturn(false);
        // Assert that the method returns false
        assertFalse(fileUtilMocks.isAgeGood(file));
    }

    @Test
    public void testIsAgeGood_FileTooSmall() throws IOException, InterruptedException {
        // Create a mock file
        File file = new File("mockFile.txt");
        file.createNewFile();
        when(fileUtilMocks.isAgeGood(file)).thenReturn(false);
        // Assert that the method returns false
        assertFalse(fileUtilMocks.isAgeGood(file));
    }
}
