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

public class FileUtil_downloadOneASINFile_0_2_Test {

    @Test
    void testDownloadOneASINFile() {
        FileUtil fileUtil = new FileUtil();
        String asin = "1234567890";
        String type = "All";
        String offer = "All";
        String page = "1";
        String saveFileName = "test.txt";
        boolean expectedDownloaded = true;
        boolean actualDownloaded = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertEquals(expectedDownloaded, actualDownloaded);
    }
}
