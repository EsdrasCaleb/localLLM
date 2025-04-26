package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;

public class a4jUtil_URLFriendlyName_0_1_Test {

    @Mock
    private LoadProperties mockLoadProperties;

    @Mock
    private Properties mockProperties;

    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        a4jUtil = new a4jUtil();
        when(mockLoadProperties.getProperties()).thenReturn(mockProperties);
        when(mockProperties.getProperty("URLSeperator")).thenReturn("-");
    }

    @Test
    public void testURLFriendlyName_withSpaces() throws Exception {
        String input = "Hello World";
        String expectedOutput = "Hello-World";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }

    @Test
    public void testURLFriendlyName_withSpecialChars() throws Exception {
        String input = "Hello!@#World$%^";
        String expectedOutput = "HelloWorld";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }

    @Test
    public void testURLFriendlyName_withAllowedChars() throws Exception {
        String input = "Hello-World_123|+~";
        String expectedOutput = "Hello-World_123|+~";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }

    @Test
    public void testURLFriendlyName_withOnlySpaces() throws Exception {
        String input = "     ";
        String expectedOutput = "-----";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }

    @Test
    public void testURLFriendlyName_withEmptyString() throws Exception {
        String input = "";
        String expectedOutput = "";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }

    @Test
    public void testURLFriendlyName_withNull() throws Exception {
        String input = null;
        String expectedOutput = "";
        assertEquals(expectedOutput, a4jUtil.URLFriendlyName(input));
    }
}
