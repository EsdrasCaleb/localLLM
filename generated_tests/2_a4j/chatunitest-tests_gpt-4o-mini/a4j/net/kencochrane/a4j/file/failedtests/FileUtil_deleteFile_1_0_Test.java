package net.kencochrane.a4j.file;

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
import java.util.Date;
import java.util.Properties;
import java.util.Random;

class FileUtil_deleteFile_1_0_Test {

    private FileUtil fileUtil;

    @BeforeEach
    void setUp() {
        fileUtil = new FileUtil();
    }

    @Test
    void testDeleteFile_FileExists_DeletedSuccessfully() {
        // Arrange
        String fileName = "testFile.txt";
        File mockFile = mock(File.class);
        when(mockFile.exists()).thenReturn(true);
        when(mockFile.delete()).thenReturn(true);
        // Use reflection to set the private field
        try {
            var field = FileUtil.class.getDeclaredField("cacheDir");
            field.setAccessible(true);
            field.set(fileUtil, mockFile.getAbsolutePath());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Act
        fileUtil.deleteFile(fileName);
        // Assert
        verify(mockFile).delete();
    }

    @Test
    void testDeleteFile_FileDoesNotExist() {
        // Arrange
        String fileName = "nonExistentFile.txt";
        File mockFile = mock(File.class);
        when(mockFile.exists()).thenReturn(false);
        // Use reflection to set the private field
        try {
            var field = FileUtil.class.getDeclaredField("cacheDir");
            field.setAccessible(true);
            field.set(fileUtil, mockFile.getAbsolutePath());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        // Act
        fileUtil.deleteFile(fileName);
        // Assert
        verify(mockFile, never()).delete();
    }

    @Test
    void testDeleteFile_FileIsNull() {
        // Arrange
        String fileName = null;
        // Act
        fileUtil.deleteFile(fileName);
        // Assert
        // No exception is expected, just ensuring method can handle null
    }
}
