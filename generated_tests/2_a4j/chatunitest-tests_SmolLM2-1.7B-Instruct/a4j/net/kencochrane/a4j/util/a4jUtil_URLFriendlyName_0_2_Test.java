// Test method
package net.kencochrane.a4j.util;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_URLFriendlyName_0_2_Test {

    @Mock
    private a4jUtil util;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testURLFriendlyName() {
        String name = "Hello World";
        String expectedResult = "Hello-World";
        when(util.URLFriendlyName(name)).thenReturn(expectedResult);
        assertEquals(expectedResult, util.URLFriendlyName(name));
    }
}
