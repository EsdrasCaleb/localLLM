package net.kencochrane.a4j.file;

import org.junit.jupiter.api.io.TempDir;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

public class FileUtil_deleteFile_1_1_Test {

    @InjectMocks
    private FileUtil fileUtil;

    @TempDir
    Path tempDir;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testDeleteFile_FileExists() throws IOException {
        // Create a temporary file
        File tempFile = Files.createFile(tempDir.resolve("testFile.txt")).toFile();
        // Call the deleteFile method
        fileUtil.deleteFile(tempFile.getAbsolutePath());
        // Verify that the file is deleted
        assertFalse(tempFile.exists());
    }

    @Test
    public void testDeleteFile_FileDoesNotExist() {
        // Call the deleteFile method with a non-existent file
        fileUtil.deleteFile("nonExistentFile.txt");
        // Since the file does not exist, the method should do nothing
        // and no exception should be thrown
    }
}
