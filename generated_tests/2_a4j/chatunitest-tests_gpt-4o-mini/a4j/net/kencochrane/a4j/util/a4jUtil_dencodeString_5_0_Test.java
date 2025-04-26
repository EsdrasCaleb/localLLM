package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
import java.net.URLDecoder;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_dencodeString_5_0_Test {

    private final a4jUtil util = new a4jUtil();

    @Test
    public void testDencodeString_ValidUTF8() throws Exception {
        String input = "Hello%20World";
        String expected = "Hello World";
        String result = invokeDencodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testDencodeString_EmptyString() throws Exception {
        String input = "";
        String expected = "";
        String result = invokeDencodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testDencodeString_NonEncodedString() throws Exception {
        String input = "Hello World";
        String expected = "Hello World";
        String result = invokeDencodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testDencodeString_UnsupportedEncoding() throws Exception {
        // Simulating an UnsupportedEncodingException by using a different encoding
        // URL encoded string for "तिती"
        String input = "%E0%A4%A4%E0%A4%BF%E0%A4%A4%E0%A4%BE";
        // Expected result
        String expected = URLDecoder.decode(input, "UTF-8");
        String result = invokeDencodeString(input);
        assertEquals(expected, result);
    }

    private String invokeDencodeString(String searchTerm) throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("dencodeString", String.class);
        method.setAccessible(true);
        return (String) method.invoke(util, searchTerm);
    }
}
