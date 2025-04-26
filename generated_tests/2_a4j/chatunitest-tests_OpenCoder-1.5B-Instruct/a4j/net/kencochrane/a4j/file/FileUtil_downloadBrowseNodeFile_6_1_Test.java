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

// Focal class
public class FileUtil_downloadBrowseNodeFile_6_1_Test {

    // Focal method
    public boolean downloadBrowseNodeFile(String mode, String node, String page, String saveFileName) {
        // log.debug("download");
        // log.debug("saveFilename = " + saveFileName);
        boolean downloaded;
        Query xml = new Query();
        String type = "lite";
        String offer = "new";
        String response = new String();
        try {
            // log.debug("download - try");
            response = xml.sendRequest(xml.browseNodeQueryGenerator(type, page, offer, mode, node));
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
        return downloaded;
    }

    @Test
    public void testDownloadBrowseNodeFile() {
        // Arrange
        String mode = "testMode";
        String node = "testNode";
        String page = "testPage";
        String saveFileName = "testSaveFileName";
        // Act
        boolean result = downloadBrowseNodeFile(mode, node, page, saveFileName);
        // Assert
        assert result == true;
    }
}
