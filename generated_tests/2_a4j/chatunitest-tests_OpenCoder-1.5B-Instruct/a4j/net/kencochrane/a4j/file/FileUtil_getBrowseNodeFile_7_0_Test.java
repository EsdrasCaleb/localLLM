package net.kencochrane.a4j.file;

import java.io.File;
import java.io.IOException;
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

public class FileUtil_getBrowseNodeFile_7_0_Test {

    private FileUtil fileUtil = new FileUtil();

    @Test
    public void testGetBrowseNodeFile() throws IOException {
        // Arrange
        String mode = "testMode";
        String node = "testNode";
        String page = "testPage";
        String cacheDir = "testCacheDir";
        long oldestAge = 0L;
        fileUtil.cacheDir = cacheDir;
        fileUtil.oldestAge = oldestAge;
        // Act
        File result = fileUtil.getBrowseNodeFile(mode, node, page);
        // Assert
        assert result != null;
        assert result.exists();
        assert result.isFile();
        assert result.getName().equals(mode + "_" + node + "_" + page + ".xml");
    }
}
