package net.kencochrane.a4j.util;

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
import java.util.Properties;

public class a4jUtil_stripString_1_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testStripString() {
        // Test case 1: All characters in string are allowed
        String allowedChars = "abc";
        String string = "abc";
        String expected = "abc";
        String result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
        // Test case 2: No characters in string are allowed
        allowedChars = "abc";
        string = "def";
        expected = "";
        result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
        // Test case 3: Some characters in string are allowed
        allowedChars = "abc";
        string = "abcd";
        expected = "abc";
        result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
        // Test case 4: Empty allowedChars
        allowedChars = "";
        string = "abc";
        expected = "";
        result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
        // Test case 5: Empty string
        allowedChars = "abc";
        string = "";
        expected = "";
        result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
        // Test case 6: Both allowedChars and string are empty
        allowedChars = "";
        string = "";
        expected = "";
        result = a4jUtil.stripString(allowedChars, string);
        assertEquals(expected, result);
    }
}
