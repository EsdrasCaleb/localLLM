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
public class FileUtil_fetchGenericSearchFile_14_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil fileUtilMock;

    @BeforeEach
    public void setUp() {
        fileUtil = spy(new FileUtil());
    }

    @Test
    public void testFetchGenericSearchFile_FileNotFound() throws Exception {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        when(fileUtil.downloadGenericSearchFile(searchType, searchTerm, mode, type, page, offer)).thenReturn(null);
        FileInputStream result = fileUtil.fetchGenericSearchFile(searchType, searchTerm, mode, type, page, offer);
        assertNull(result);
        verify(fileUtil, times(1)).downloadGenericSearchFile(searchType, searchTerm, mode, type, page, offer);
    }
}
