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

public class FileUtil_renameFile_3_0_Test {

    @Test
    public void testRenameFile() {
        // Arrange
        FileUtil fileUtil = new FileUtil();
        String oldFileName = "old.txt";
        String newFileName = "new.txt";
        // Act
        fileUtil.renameFile(oldFileName, newFileName);
        // Assert
        assertTrue(new File(newFileName).exists());
        assertFalse(new File(oldFileName).exists());
    }
}
