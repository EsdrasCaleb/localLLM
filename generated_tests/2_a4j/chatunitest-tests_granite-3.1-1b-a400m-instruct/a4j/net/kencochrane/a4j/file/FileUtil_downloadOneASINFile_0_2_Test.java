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

    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() {
        fileUtil = new FileUtil();
    }

    @Test
    public void testDownloadOneASINFile() {
        String asin = "ASIN1234567890";
        String type = "ASINSearch";
        String offer = "All";
        String page = "1";
        String saveFileName = "file.txt";
        boolean downloaded = fileUtil.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertTrue(downloaded);
        // Cleanup: remove the downloaded file
        try {
            if (saveFileName.endsWith(".txt")) {
                File file = new File(saveFileName);
                file.delete();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
