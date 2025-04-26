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
public class FileUtil_downloadOneASINFile_0_3_Test {

    // Test class
    @Test
    public void testDownloadOneASINFile() {
        FileUtil fileUtil = new FileUtil();
        fileUtil.cacheDir = "C:\\Users\\User\\Desktop\\";
        fileUtil.oldestAge = 1000;
        boolean result = fileUtil.downloadOneASINFile("B000000000", "books", "all", "1", "file1.txt");
        assertEquals(true, result);
    }
}
