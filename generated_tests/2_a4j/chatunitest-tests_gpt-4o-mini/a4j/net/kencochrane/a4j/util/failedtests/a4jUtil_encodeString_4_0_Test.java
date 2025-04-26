package net.kencochrane.a4j.util;

import java.io.UnsupportedEncodingException;
import java.lang.reflect.Method;
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

public class a4jUtil_encodeString_4_0_Test {

    private final a4jUtil util = new a4jUtil();

    @Test
    public void testEncodeString_ValidInput() throws Exception {
        String input = "Hello World!";
        // Expected URL-encoded output
        String expected = "Hello%20World%21";
        String result = util.encodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testEncodeString_EmptyInput() throws Exception {
        String input = "";
        // Expected URL-encoded output for empty string
        String expected = "";
        String result = util.encodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testEncodeString_NullInput() throws Exception {
        String input = null;
        // Expected output for null input
        String expected = null;
        String result = util.encodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testEncodeString_SpecialCharacters() throws Exception {
        String input = "!@#$%^&*()";
        // Expected URL-encoded output
        String expected = "%21%40%23%24%25%5E%26%2A%28%29";
        String result = util.encodeString(input);
        assertEquals(expected, result);
    }

    @Test
    public void testEncodeString_UnsupportedEncoding() throws Exception {
        Method method = a4jUtil.class.getDeclaredMethod("encodeString", String.class);
        method.setAccessible(true);
        // Simulate UnsupportedEncodingException by using reflection if necessary
        // This is just an example, as URLEncoder.encode should not throw this for UTF-8
        // We can only test the fallback if we had a different scenario, but here it's safe.
        String input = "Test";
        // Expected fallback output (not realistic in this case)
        String expected = "Test";
        String result = (String) method.invoke(util, input);
        assertEquals(expected, result);
    }
}
