package net.kencochrane.a4j.file;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
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
class FileUtil_deleteFile_1_0_Test {

    @Mock
    private File file;

    @InjectMocks
    private FileUtil fileUtil;

    @Test
    void deleteFileTest() {
        // Arrange
        String fileName = "test.txt";
        when(file.exists()).thenReturn(true);
        when(file.delete()).thenReturn(true);
        // Act
        fileUtil.deleteFile(fileName);
        // Assert
        assertTrue(file.delete());
    }
}
