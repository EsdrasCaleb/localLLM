package net.kencochrane.a4j.util;

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

public class a4jUtil_URLFriendlyName_0_0_Test {

    @Mock
    private LoadProperties loadProperties;

    private a4jUtil util;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        util = new a4jUtil();
        // Mock the properties
        Properties props = new Properties();
        props.setProperty("URLSeperator", "-");
        when(loadProperties.getProperties()).thenReturn(props);
        // Use reflection to set the mocked LoadProperties instance
        try {
            java.lang.reflect.Field field = a4jUtil.class.getDeclaredField("loadProperties");
            field.setAccessible(true);
            field.set(null, loadProperties);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testURLFriendlyName_WithSpaces() {
        String result = util.URLFriendlyName("Hello World");
        assertEquals("Hello-World", result);
    }

    @Test
    public void testURLFriendlyName_WithAllowedChars() {
        String result = util.URLFriendlyName("Hello_123");
        assertEquals("Hello_123", result);
    }

    @Test
    public void testURLFriendlyName_WithDisallowedChars() {
        String result = util.URLFriendlyName("Hello @ World!");
        assertEquals("Hello-World", result);
    }

    @Test
    public void testURLFriendlyName_EmptyString() {
        String result = util.URLFriendlyName("");
        assertEquals("", result);
    }

    @Test
    public void testURLFriendlyName_NullInput() {
        String result = util.URLFriendlyName(null);
        // Assuming the method handles null by returning an empty string
        assertEquals("", result);
    }
}
