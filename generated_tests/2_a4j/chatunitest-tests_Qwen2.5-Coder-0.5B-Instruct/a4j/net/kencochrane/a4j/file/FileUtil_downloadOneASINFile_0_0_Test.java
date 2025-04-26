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

class FileUtil_downloadOneASINFile_0_0_Test {

    @Test
    public void testDownloadOneASINFile() throws Exception {
        FileUtil util = new FileUtil();
        String asin = "B000000001";
        String type = "Books";
        String offer = "all";
        String page = "1";
        String saveFileName = "downloaded_file.txt";
        boolean downloaded = util.downloadOneASINFile(asin, type, offer, page, saveFileName);
        assertTrue(downloaded);
        assertEquals("downloaded_file.txt", util.cacheDir);
    }
}
