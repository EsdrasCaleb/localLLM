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

@ExtendWith(MockitoExtension.class)
public class FileUtil_downloadOneASINFile_0_1_Test {

    // Test class
    @Test
    public void testDownloadOneASINFile() {
        // Arrange
        FileUtil fileUtil = new FileUtil();
        String asin = "0123456789";
        String type = "books";
        String offer = "all";
        String page = "1";
        String saveFileName = "testFile.txt";
        // Act
        boolean downloaded = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        // Assert
        assertTrue(downloaded);
    }
}
