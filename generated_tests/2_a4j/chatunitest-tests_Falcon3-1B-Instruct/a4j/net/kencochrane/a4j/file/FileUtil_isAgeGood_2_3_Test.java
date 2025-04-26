package net.kencochrane.a4j.file;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_isAgeGood_2_3_Test {

    @Test
    public void testIsAgeGood() {
        // Arrange
        FileUtil fileUtil = new FileUtil();
        // Act
        File file = new File("example.txt", "example.txt");
        boolean result = fileUtil.isAgeGood(file);
        // Assert
        assertEquals(false, result);
    }
}
