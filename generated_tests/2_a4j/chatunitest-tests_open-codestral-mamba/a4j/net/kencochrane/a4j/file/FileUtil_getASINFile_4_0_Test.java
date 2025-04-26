package net.kencochrane.a4j.file;

import java.io.File;
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

class FileUtil_getASINFile_4_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil testFileUtil;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetASINFile() {
        String asin = "B01M7ZN4NH";
        String type = "testType";
        String offer = "testOffer";
        String page = "testPage";
        when(fileUtil.getASINFile(asin, type, offer, page)).thenReturn(new File("testFilePath"));
        File result = testFileUtil.getASINFile(asin, type, offer, page);
        assertNotNull(result);
        assertEquals("testFilePath", result.getPath());
    }
}
