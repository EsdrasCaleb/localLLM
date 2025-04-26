package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.UnsupportedEncodingException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_encodeString_4_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @Test
    public void testEncodeString_WithValidString_ReturnsEncodedString() {
        String searchTerm = "Hello World";
        String expected = "Hello%20World";
        String actual = a4jUtil.encodeString(searchTerm);
        assertEquals(expected, actual);
    }

    @Test
    public void testEncodeString_WithInvalidString_ThrowsUnsupportedEncodingException() {
        String searchTerm = "Hello World";
        assertThrows(UnsupportedEncodingException.class, () -> a4jUtil.encodeString(searchTerm));
    }

    @Test
    public void testEncodeString_WithNullInput_ThrowsNullPointerException() {
        assertThrows(NullPointerException.class, () -> a4jUtil.encodeString(null));
    }
}
