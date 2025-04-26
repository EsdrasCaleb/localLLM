package net.kencochrane.a4j.file;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_fetchASINFile_5_4_Test {

    @Test
    public void testFetchASINFile() throws Exception {
        // Arrange
        String asin = "123456789012";
        String type = "image";
        String offer = "premium";
        String page = "1";
        FileUtil fileUtil = new FileUtil();
        // Act
        FileInputStream fileInputStream = fileUtil.fetchASINFile(asin, type, offer, page);
        // Assert
        assertEquals(asin, "123456789012");
        assertEquals(type, "image");
        assertEquals(offer, "premium");
        assertEquals(page, "1");
        assertNull(fileInputStream);
    }
}
