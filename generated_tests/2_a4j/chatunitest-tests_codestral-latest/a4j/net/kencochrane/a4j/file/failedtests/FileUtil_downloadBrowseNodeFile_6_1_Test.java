package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
public class FileUtil_downloadBrowseNodeFile_6_1_Test {

    @Mock
    private Query mockQuery;

    @InjectMocks
    private FileUtil fileUtil;

    @BeforeEach
    public void setUp() throws Exception {
        Field queryField = FileUtil.class.getDeclaredField("query");
        queryField.setAccessible(true);
        queryField.set(fileUtil, mockQuery);
    }

    @Test
    public void testDownloadBrowseNodeFile_Success() throws Exception {
        String mode = "mode";
        String node = "node";
        String page = "page";
        String saveFileName = "testFile.txt";
    }
}
