// Test method
package net.kencochrane.a4j.file;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.util.LoadProperties;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.Properties;
import java.util.Random;
import java.util.stream.Collectors;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class FileUtil_getASINFile_4_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private FileUtil fileUtilUnderTest;

    @Test
    public void testGetASINFile() {
        // Arrange
        String asin = "B001234567";
        String type = "ebook";
        String offer = "10.99";
        String page = "1";
        // Act
        File file = fileUtilUnderTest.getASINFile(asin, type, offer, page);
        // <Buggy Line>: reference to assertNotNull is ambiguous  both method assertNotNull(java.lang.Object) in org.junit.Assert and method assertNotNull(java.lang.Object) in org.junit.jupiter.api.Assertions match
        assertNotNull(file);
    }
}
