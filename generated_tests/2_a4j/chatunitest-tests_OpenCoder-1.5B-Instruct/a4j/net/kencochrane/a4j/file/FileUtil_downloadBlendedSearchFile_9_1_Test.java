package net.kencochrane.a4j.file;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;

class FileUtil_downloadBlendedSearchFile_9_1_Test {

    private String cacheDir;

    private long oldestAge;

    public File downloadBlendedSearchFile(String searchTerm, String type) {
        // log.debug("download");
        Date timestamp = new Date();
        Random r = new Random();
        String fileName = "b_" + Long.toString(timestamp.getTime()) + "_" + r.nextInt() + ".xml";
        String saveFileName = cacheDir + fileName.trim().toUpperCase();
        // log.debug("saveFilename = " + saveFileName);
        boolean downloaded;
        Query xml = new Query();
        String response = new String();
        try {
            // log.debug("download - try");
            response = xml.sendRequest(xml.BlendedSearchGenerator(type, searchTerm));
            FileOutputStream out = new FileOutputStream(saveFileName);
            byte[] byteMe = response.getBytes();
            out.write(byteMe);
            out.close();
            File file = new File(saveFileName);
            if (file != null && file.length() < 1000) {
                // log.debug("FileSize = " + file.length());
                downloaded = false;
            } else {
                downloaded = true;
            }
        } catch (Exception e) {
            // log.error(e.toString());
            downloaded = false;
        }
        if (downloaded) {
            return new File(saveFileName);
        } else {
            return null;
        }
    }
}

class FileUtilTest {

    private static final String CACHE_DIR = "test_cache/";

    private static final long OLDEST_AGE = 1000;

    @Test
    public void testDownloadBlendedSearchFile() {
        // Setup
        String searchTerm = "test_search_term";
        FileUtil_downloadBlendedSearchFile_9_1_Test mockFileUtil = spy(new FileUtil_downloadBlendedSearchFile_9_1_Test());
        // Run the method under test
        File downloadedFile = mockFileUtil.downloadBlendedSearchFile(searchTerm, "test_type");
        // Verify the result
        assertNotNull(downloadedFile);
        assertTrue(downloadedFile.exists());
        assertEquals(CACHE_DIR + "B_" + System.currentTimeMillis() + "_" + new Random().nextInt() + ".xml", downloadedFile.getAbsolutePath());
    }
}
