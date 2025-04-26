package net.kencochrane.a4j.file;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;

@ExtendWith(MockitoExtension.class)
public class FileUtil_fetchKeywordSearchFile_12_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil fileUtilMock;

    @BeforeEach
    public void setUp() {
        fileUtil = spy(new FileUtil());
    }

    @Test
    public void testFetchKeywordSearchFile_FileNotFound() throws FileNotFoundException {
        String searchTerm = "term";
        String productLine = "line";
        String type = "type";
        String page = "page";
        when(fileUtil.downloadKeywordSearchFile(searchTerm, productLine, type, page)).thenReturn(null);
        FileInputStream result = fileUtil.fetchKeywordSearchFile(searchTerm, productLine, type, page);
        assertNull(result);
        verify(fileUtil, times(1)).downloadKeywordSearchFile(searchTerm, productLine, type, page);
    }
}
