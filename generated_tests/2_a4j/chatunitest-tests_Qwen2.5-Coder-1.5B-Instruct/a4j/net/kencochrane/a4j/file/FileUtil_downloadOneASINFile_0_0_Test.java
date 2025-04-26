package net.kencochrane.a4j.file;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.Date;
import java.util.Properties;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class FileUtil_downloadOneASINFile_0_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil downloadOneASINFile;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testDownloadOneASINFile() throws IOException {
        // Arrange
        when(fileUtil.downloadOneASINFile(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn(true);
        // Act
        boolean result = downloadOneASINFile.downloadOneASINFile("1234567890", "type", "offer", "page", "saveFileName");
        // Assert
        assertTrue(result);
        verify(fileUtil).downloadOneASINFile("1234567890", "type", "offer", "page", "saveFileName");
    }
}
